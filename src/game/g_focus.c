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
// g_focus.c -- hold-to-focus world time dilation (player movement stays full speed)

#include "g_local.h"

#define FOCUS_TIMESCALE		0.125f	/* world/effects run at 1/8 speed */
#define FOCUS_RAMP			4.0f	/* ~250ms in and out at 10 Hz */
#define FOCUS_CHARGE_TIME	30.0f	/* seconds to fill an empty battery */
#define FOCUS_DRAIN_TIME	15.0f	/* seconds of focus from a full battery */

static cvar_t *g_focus;
static cvar_t *g_focus_scale;

void G_InitFocus(void)
{
	g_focus = gi.cvar("g_focus", "1", 0);
	g_focus_scale = gi.cvar("g_focus_scale", "0.125", 0);
	if (g_focus_scale->value >= 0.24f && g_focus_scale->value <= 0.29f)
		g_focus_scale = gi.cvar_set("g_focus_scale", "0.125");
	level.frametime = FRAMETIME;
	level.focus_scale = 1.0f;
	level.focus_frac = 0.0f;
}

static void G_RunFocusBattery(edict_t *ent)
{
	gclient_t *client = ent->client;
	float dt = FRAMETIME;

	if (!client)
		return;

	if (ent->health <= 0 || client->resp.spectator || level.intermissiontime) {
		client->focus_active = false;
		return;
	}

	if (client->focus_active) {
		if (!client->focus_held) {
			client->focus_active = false;
		} else {
			client->pers.focus_battery -= dt / FOCUS_DRAIN_TIME;
			if (client->pers.focus_battery <= 0.0f) {
				client->pers.focus_battery = 0.0f;
				client->focus_active = false;
			}
		}
	}

	if (!client->focus_active) {
		client->pers.focus_battery += dt / FOCUS_CHARGE_TIME;
		if (client->pers.focus_battery > 1.0f)
			client->pers.focus_battery = 1.0f;

		if (g_focus && g_focus->value && client->focus_held && client->pers.focus_battery >= 1.0f)
			client->focus_active = true;
	}
}

qboolean G_FocusEnabled(void)
{
	return g_focus && g_focus->value;
}

int G_FocusHudSeconds(gclient_t *client)
{
	float battery;
	int seconds;

	if (!client)
		return 0;

	battery = client->pers.focus_battery;
	if (client->focus_active)
		seconds = (int)(battery * FOCUS_DRAIN_TIME + 0.999f);
	else if (battery >= 1.0f)
		seconds = (int)FOCUS_DRAIN_TIME;
	else
		seconds = (int)((1.0f - battery) * FOCUS_CHARGE_TIME + 0.999f);

	if (seconds < 0)
		seconds = 0;
	else if (seconds > 99)
		seconds = 99;
	return seconds;
}

void G_UpdateFocus(void)
{
	qboolean wanted = false;
	float target, step, scale;
	int i;
	edict_t *ent;

	if (!g_focus)
		G_InitFocus();

	for (i = 1; i <= game.maxclients; i++) {
		ent = &g_edicts[i];
		if (!ent->inuse || !ent->client)
			continue;
		G_RunFocusBattery(ent);
		if (g_focus->value && !level.intermissiontime && ent->health > 0
			&& !ent->client->resp.spectator && ent->client->focus_active)
			wanted = true;
	}

	target = wanted ? 1.0f : 0.0f;
	step = FRAMETIME * FOCUS_RAMP;
	if (level.focus_frac < target) {
		level.focus_frac += step;
		if (level.focus_frac > target)
			level.focus_frac = target;
	} else if (level.focus_frac > target) {
		level.focus_frac -= step;
		if (level.focus_frac < target)
			level.focus_frac = target;
	}

	scale = g_focus_scale ? g_focus_scale->value : FOCUS_TIMESCALE;
	if (scale < 0.05f)
		scale = 0.05f;
	else if (scale > 1.0f)
		scale = 1.0f;

	level.focus_scale = 1.0f - level.focus_frac * (1.0f - scale);
	level.frametime = FRAMETIME * level.focus_scale;
}
