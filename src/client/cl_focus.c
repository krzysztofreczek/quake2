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
// cl_focus.c -- hold-to-focus client ramp and dilated effect clock

#include "client.h"

#define FOCUS_TIMESCALE		0.25f
#define FOCUS_FOV			0.88f	/* slight FOV tighten while focused */
#define FOCUS_RAMP			4.0f	/* ~250ms in and out */

float		cl_focus_time;
float		cl_focus_scale = 1.0f;
float		cl_focus_frac;

cvar_t		*cl_focus;
cvar_t		*cl_focus_scale_cvar;
cvar_t		*cl_focus_fov;

extern kbutton_t	in_focus;

static int		focus_wallclock;

static qboolean Focus_IsHeld(void)
{
	if (!cl_focus || !cl_focus->integer)
		return false;
	if (cls.state != ca_active || cls.key_dest != key_game)
		return false;
	if (cl_paused->integer)
		return false;
	if (!cl.frame.valid)
		return false;
	if (cl.frame.playerstate.pmove.pm_type != PM_NORMAL)
		return false;
	return (in_focus.state & 1) != 0;
}

static float Focus_BlendToward(float current, float wanted, float rate)
{
	float delta = wanted - current;
	float step = cls.frametime * rate;

	if (delta > step)
		return current + step;
	if (delta < -step)
		return current - step;
	return wanted;
}

void CL_FocusReset(void)
{
	cl_focus_frac = 0.0f;
	cl_focus_scale = 1.0f;
	cl_focus_time = (float)cl.time;
	focus_wallclock = cl.time;
}

void CL_UpdateFocus(void)
{
	float target, scale;
	int now;

	if (!cl_focus)
		return;

	target = Focus_IsHeld() ? 1.0f : 0.0f;
	cl_focus_frac = Focus_BlendToward(cl_focus_frac, target, FOCUS_RAMP);

	scale = cl_focus_scale_cvar ? cl_focus_scale_cvar->value : FOCUS_TIMESCALE;
	if (scale < 0.05f)
		scale = 0.05f;
	else if (scale > 1.0f)
		scale = 1.0f;

	cl_focus_scale = 1.0f - cl_focus_frac * (1.0f - scale);

	now = cl.time;
	if (focus_wallclock <= 0 || now < focus_wallclock || now - focus_wallclock > 500) {
		cl_focus_time = (float)now;
	} else {
		cl_focus_time += (float)(now - focus_wallclock) * cl_focus_scale;
	}
	focus_wallclock = now;
}

void CL_FocusApplyView(void)
{
	float fov_scale;

	if (cl_focus_frac <= 0.0f)
		return;

	fov_scale = cl_focus_fov ? cl_focus_fov->value : FOCUS_FOV;
	if (fov_scale < 0.7f)
		fov_scale = 0.7f;
	else if (fov_scale > 1.0f)
		fov_scale = 1.0f;

	cl.refdef.fov_x *= 1.0f - cl_focus_frac * (1.0f - fov_scale);
}

void CL_InitFocus(void)
{
	const char *b;

	cl_focus = Cvar_Get("cl_focus", "1", CVAR_ARCHIVE);
	cl_focus_scale_cvar = Cvar_Get("cl_focus_scale", "0.25", 0);
	cl_focus_fov = Cvar_Get("cl_focus_fov", "0.88", CVAR_ARCHIVE);

	if (cl_focus_scale_cvar && cl_focus_scale_cvar->value > 0.27f && cl_focus_scale_cvar->value < 0.29f)
		Cvar_Set("cl_focus_scale", "0.25");

	CL_FocusReset();

	/* Default to X. Drop leftover F binds from the previous default. */
	if (keys['f'].binding && !Q_stricmp(keys['f'].binding, "+focus"))
		Key_SetBinding('f', "");

	{
		int i;
		qboolean bound = false;

		for (i = 0; i < MAX_KEYS; i++) {
			b = keys[i].binding;
			if (b && !Q_stricmp(b, "+focus")) {
				bound = true;
				break;
			}
		}
		if (!bound)
			Key_SetBinding('x', "+focus");
	}
}
