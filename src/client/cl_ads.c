/*
Copyright (C) 1997-2001 Id Software, Inc.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

*/
// cl_ads.c -- hold-to-ADS view, centered weapon pose, and sticky auto-aim

#include "client.h"

/*
 * Modern ADS (hold right-click):
 *  - Weapon eases to a centered, unsided pose
 *  - FOV tightens so the view feels focused on the weapon
 *  - If an enemy is on-screen, aim magnetizes toward center-mass
 *  - Releasing the button eases weapon, FOV, and assist back to hip-fire
 */

#define ADS_BLEND_RATE		14.0f	/* ~70ms in/out */
#define ADS_GUN_Y			-4.0f	/* pull viewmodel toward the camera */
#define ADS_GUN_Z			1.0f	/* slight raise into sight line */
#define ADS_VIEW_PUSH		6.0f	/* lean the eye slightly into the aim */
#define ADS_BOB_DAMP		0.85f
#define ADS_LOOK_ADS		0.90f	/* extra tightness on top of FOV scale */
#define ADS_ACQUIRE_DEG		38.0f	/* enemies clearly in view */
#define ADS_STICKY_DEG		48.0f	/* keep lock a bit past acquire */
#define ADS_MAX_RANGE		1500.0f
#define ADS_ACQUIRE_TAU		0.05f
#define ADS_TRACK_TAU		0.08f
#define ADS_LOCK_MS			100
#define ADS_DEADZONE		0.35f
#define ADS_FOV_DEFAULT		0.70f

float		cl_ads_frac;

cvar_t		*cl_ads;
cvar_t		*cl_ads_fov;
cvar_t		*cl_ads_autoaim;

extern kbutton_t	in_ads;

static int		ads_target = -1;
static unsigned	ads_lock_time;

static float Ads_AngleDiff(float a, float b)
{
	float d = a - b;

	while (d > 180.0f)
		d -= 360.0f;
	while (d < -180.0f)
		d += 360.0f;
	return d;
}

static float Ads_BlendToward(float current, float wanted, float rate)
{
	float delta = wanted - current;
	float step = cls.frametime * rate;

	if (delta > step)
		return current + step;
	if (delta < -step)
		return current - step;
	return wanted;
}

static void Ads_CurrentAngles(vec3_t angles)
{
	angles[PITCH] = cl.viewangles[PITCH] + SHORT2ANGLE(cl.frame.playerstate.pmove.delta_angles[PITCH]);
	angles[YAW] = cl.viewangles[YAW] + SHORT2ANGLE(cl.frame.playerstate.pmove.delta_angles[YAW]);
	angles[ROLL] = cl.viewangles[ROLL] + SHORT2ANGLE(cl.frame.playerstate.pmove.delta_angles[ROLL]);
}

static qboolean Ads_IsHeld(void)
{
	if (!cl_ads || !cl_ads->integer)
		return false;
	if (cls.state != ca_active || cls.key_dest != key_game)
		return false;
	if (cl_paused->integer)
		return false;
	if (!cl.frame.valid)
		return false;
	if (cl.frame.playerstate.pmove.pm_type != PM_NORMAL)
		return false;
	return (in_ads.state & 1) != 0;
}

static qboolean Ads_IsEnemy(const entity_state_t *ent)
{
	const char *model;

	if (!ent->solid || !ent->modelindex)
		return false;
	if (ent->number == cl.playernum + 1)
		return false;
	if (ent->effects & (EF_GIB | EF_GREENGIB | EF_ROTATE))
		return false;
	if (ent->solid == 31)
		return false;

	if (ent->number >= 1 && ent->number <= cl.maxclients)
		return true;
	if (ent->modelindex == 255)
		return true;

	model = cl.configstrings[CS_MODELS + ent->modelindex];
	if (!model || !model[0] || model[0] == '*')
		return false;

	return strstr(model, "models/monsters/") != NULL;
}

static void Ads_AimPoint(const entity_state_t *ent, vec3_t out)
{
	const centity_t *cent = &cl_entities[ent->number];
	int i, zu;
	float z_off = 22.0f;

	for (i = 0; i < 3; i++)
		out[i] = cent->prev.origin[i] + cl.lerpfrac * (cent->current.origin[i] - cent->prev.origin[i]);

	if (ent->solid && ent->solid != 31) {
		if (cls.protocolVersion >= PROTOCOL_VERSION_R1Q2_SOLID)
			zu = ((ent->solid >> 16) & 65535) - 32768;
		else
			zu = 8 * ((ent->solid >> 10) & 63) - 32;
		z_off = (float)zu * 0.55f;
	}

	out[2] += z_off;
}

static qboolean Ads_HasLOS(const vec3_t start, const vec3_t end)
{
	trace_t tr;

	tr = CM_BoxTrace(start, end, vec3_origin, vec3_origin, 0, MASK_SOLID);
	return tr.fraction >= 1.0f;
}

static qboolean Ads_FindTarget(vec3_t out_angles, int *out_ent)
{
	int i;
	float best_score = 999999.0f;
	int best_ent = -1;
	vec3_t best_angles, eye, forward, look;

	Ads_CurrentAngles(look);
	VectorCopy(cl.predicted_origin, eye);
	VectorAdd(eye, cl.frame.playerstate.viewoffset, eye);
	AngleVectors(look, forward, NULL, NULL);

	for (i = 0; i < cl.frame.num_entities; i++) {
		entity_state_t *ent = &cl_parse_entities[(cl.frame.parse_entities + i) & PARSE_ENTITIES_MASK];
		vec3_t aim, dir, angles;
		float dist, yaw_diff, pitch_diff, ang, score, max_ang;

		if (!Ads_IsEnemy(ent))
			continue;

		Ads_AimPoint(ent, aim);
		VectorSubtract(aim, eye, dir);
		dist = VectorNormalize(dir);
		if (dist < 16.0f || dist > ADS_MAX_RANGE)
			continue;
		if (DotProduct(dir, forward) <= 0.0f)
			continue;

		VecToAngles(dir, angles);
		yaw_diff = Ads_AngleDiff(angles[YAW], look[YAW]);
		pitch_diff = Ads_AngleDiff(angles[PITCH], look[PITCH]);
		ang = (float)sqrt(yaw_diff * yaw_diff + pitch_diff * pitch_diff);

		max_ang = (ent->number == ads_target) ? ADS_STICKY_DEG : ADS_ACQUIRE_DEG;
		if (ang > max_ang)
			continue;
		if (!Ads_HasLOS(eye, aim))
			continue;

		/* Prefer the enemy nearest the crosshair; break ties by range. */
		score = ang * 40.0f + dist * 0.02f;
		if (ent->number == ads_target)
			score -= 8.0f;
		if (score < best_score) {
			best_score = score;
			best_ent = ent->number;
			VectorCopy(angles, best_angles);
		}
	}

	if (best_ent < 0)
		return false;

	VectorCopy(best_angles, out_angles);
	*out_ent = best_ent;
	return true;
}

static void Ads_ApplyAutoAim(void)
{
	vec3_t target_angles, look;
	int ent_num = -1;
	float pitch_diff, yaw_diff, ang, tau, k;

	if (!cl_ads_autoaim->integer || cl_ads_frac <= 0.01f)
		return;

	if (!Ads_FindTarget(target_angles, &ent_num)) {
		ads_target = -1;
		return;
	}

	if (ent_num != ads_target) {
		ads_target = ent_num;
		ads_lock_time = cls.realtime;
	}

	Ads_CurrentAngles(look);
	pitch_diff = Ads_AngleDiff(target_angles[PITCH], look[PITCH]);
	yaw_diff = Ads_AngleDiff(target_angles[YAW], look[YAW]);
	ang = (float)sqrt(pitch_diff * pitch_diff + yaw_diff * yaw_diff);
	if (ang < ADS_DEADZONE)
		return;

	tau = (cls.realtime - ads_lock_time < ADS_LOCK_MS) ? ADS_ACQUIRE_TAU : ADS_TRACK_TAU;
	k = 1.0f - (float)exp(-cls.frametime / tau);
	k *= cl_ads_frac;

	cl.viewangles[PITCH] += pitch_diff * k;
	cl.viewangles[YAW] += yaw_diff * k;
}

void CL_AdsReset(void)
{
	cl_ads_frac = 0.0f;
	ads_target = -1;
	ads_lock_time = 0;
}

float CL_AdsLookScale(void)
{
	float fov_scale;

	if (cl_ads_frac <= 0.0f || !cl_ads_fov)
		return 1.0f;

	fov_scale = cl_ads_fov->value;
	if (fov_scale < 0.5f)
		fov_scale = 0.5f;
	else if (fov_scale > 1.0f)
		fov_scale = 1.0f;

	return 1.0f - cl_ads_frac * (1.0f - fov_scale * ADS_LOOK_ADS);
}

void CL_AdsApplyView(void)
{
	float fov_scale;

	if (cl_ads_frac <= 0.0f)
		return;

	VectorMA(cl.refdef.vieworg, cl_ads_frac * ADS_VIEW_PUSH, cl.v_forward, cl.refdef.vieworg);

	fov_scale = cl_ads_fov ? cl_ads_fov->value : ADS_FOV_DEFAULT;
	if (fov_scale < 0.5f)
		fov_scale = 0.5f;
	else if (fov_scale > 1.0f)
		fov_scale = 1.0f;

	cl.refdef.fov_x *= 1.0f - cl_ads_frac * (1.0f - fov_scale);
}

void CL_AdsGunOffset(float base_x, float base_y, float base_z, float *x, float *y, float *z)
{
	float frac = cl_ads_frac;

	if (frac <= 0.0f) {
		*x = base_x;
		*y = base_y;
		*z = base_z;
		return;
	}

	/* Center horizontally and pull the same unsided model toward the camera. */
	*x = base_x * (1.0f - frac);
	*y = base_y + (ADS_GUN_Y - base_y) * frac;
	*z = base_z + (ADS_GUN_Z - base_z) * frac;
}

float CL_AdsBobScale(void)
{
	return 1.0f - ADS_BOB_DAMP * cl_ads_frac;
}

qboolean CL_AdsForceCenterWeapon(void)
{
	return cl_ads_frac > 0.01f;
}

void CL_UpdateAds(void)
{
	float wanted;

	if (!cl_ads) {
		cl_ads_frac = 0.0f;
		return;
	}

	wanted = Ads_IsHeld() ? 1.0f : 0.0f;
	cl_ads_frac = Ads_BlendToward(cl_ads_frac, wanted, ADS_BLEND_RATE);
	if (cl_ads_frac < 0.001f)
		cl_ads_frac = 0.0f;
	else if (cl_ads_frac > 0.999f)
		cl_ads_frac = 1.0f;

	if (wanted <= 0.0f)
		ads_target = -1;
	else
		Ads_ApplyAutoAim();
}

void CL_InitAds(void)
{
	cvar_t *fov;
	int i;

	cl_ads = Cvar_Get("cl_ads", "1", CVAR_ARCHIVE);
	cl_ads_fov = Cvar_Get("cl_ads_fov", "0.70", CVAR_ARCHIVE);
	cl_ads_autoaim = Cvar_Get("cl_ads_autoaim", "1", CVAR_ARCHIVE);

	CL_AdsReset();

	/* The old prototype wrote a zoomed fov into the archived cvar. */
	fov = Cvar_Get("fov", "90", CVAR_USERINFO | CVAR_ARCHIVE);
	if (fov->value < 50.0f || fov->value > 160.0f)
		Cvar_Set("fov", "90");

	/* Migrate leftover +centerweapon binds from the old prototype. */
	for (i = 0; i < MAX_KEYS; i++) {
		if (keys[i].binding && !Q_stricmp(keys[i].binding, "+centerweapon"))
			Key_SetBinding(i, "+ads");
	}

	/* default.cfg often binds mouse2 to +moveup; take RMB for ADS unless customized. */
	{
		const char *b = keys[K_MOUSE2].binding;
		if (!b || !b[0] || !Q_stricmp(b, "+moveup") || !Q_stricmp(b, "+strafe"))
			Key_SetBinding(K_MOUSE2, "+ads");
	}
}
