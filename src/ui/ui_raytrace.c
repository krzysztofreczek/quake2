/*
 * Raytracing settings menu.
 *
 * Applies immediately (no vid restart). Values are archived cvars.
 */
#include "ui_local.h"

static menuframework_s	s_rt_menu;

static menulist_s		s_rt_enable_box;
static menulist_s		s_rt_mode_box;
static menulist_s		s_rt_quality_box;
static menulist_s		s_rt_bounces_box;
static menulist_s		s_rt_accum_box;
static menuslider_s		s_rt_exposure_slider;
static menuslider_s		s_rt_lightscale_slider;
static menuslider_s		s_rt_skylight_slider;
static menuslider_s		s_rt_reflect_slider;
static menulist_s		s_rt_threads_box;
static menulist_s		s_rt_weapons_box;
static menuaction_s		s_rt_defaults_action;

static const int rt_quality_scales[] = { 8, 6, 4, 2, 1 };
static const int rt_thread_counts[] = { 0, 1, 2, 4, 8 };

extern void Cvar_SetDefault(const char *var_name);

static float RT_Clamp(float minv, float maxv, float v)
{
	if (v < minv)
		return minv;
	if (v > maxv)
		return maxv;
	return v;
}

static int RT_QualityFromScale(int scale)
{
	if (scale <= 1)
		return 4;
	if (scale <= 2)
		return 3;
	if (scale <= 4)
		return 2;
	if (scale <= 6)
		return 1;
	return 0;
}

static int RT_ThreadsIndex(int threads)
{
	int i;

	if (threads <= 0)
		return 0;
	for (i = 1; i < 5; i++) {
		if (threads <= rt_thread_counts[i])
			return i;
	}
	return 4;
}

static void RT_EnsureCvars(void)
{
	Cvar_Get("r_raytrace", "1", CVAR_ARCHIVE);
	Cvar_Get("rt_mode", "0", CVAR_ARCHIVE);
	Cvar_Get("rt_scale", "4", CVAR_ARCHIVE);
	Cvar_Get("rt_bounces", "1", CVAR_ARCHIVE);
	Cvar_Get("rt_accum", "1", CVAR_ARCHIVE);
	Cvar_Get("rt_exposure", "2.6", CVAR_ARCHIVE);
	Cvar_Get("rt_lightscale", "1.35", CVAR_ARCHIVE);
	Cvar_Get("rt_skylight", "2.4", CVAR_ARCHIVE);
	Cvar_Get("rt_reflect", "0.42", CVAR_ARCHIVE);
	Cvar_Get("rt_threads", "0", CVAR_ARCHIVE);
	Cvar_Get("rt_weapons", "1", CVAR_ARCHIVE);
}

static void RT_SetMenuItemValues(void)
{
	int scale, threads;

	RT_EnsureCvars();

	s_rt_enable_box.curvalue = Cvar_VariableIntValue("r_raytrace") != 0;
	s_rt_mode_box.curvalue = Cvar_VariableIntValue("rt_mode") != 0;

	scale = Cvar_VariableIntValue("rt_scale");
	s_rt_quality_box.curvalue = RT_QualityFromScale(scale);

	s_rt_bounces_box.curvalue = Cvar_VariableIntValue("rt_bounces") != 0;
	s_rt_accum_box.curvalue = Cvar_VariableIntValue("rt_accum") != 0;

	s_rt_exposure_slider.curvalue = RT_Clamp(8, 50, Cvar_VariableValue("rt_exposure") * 10.0f);
	s_rt_lightscale_slider.curvalue = RT_Clamp(5, 30, Cvar_VariableValue("rt_lightscale") * 10.0f);
	s_rt_skylight_slider.curvalue = RT_Clamp(0, 50, Cvar_VariableValue("rt_skylight") * 10.0f);
	s_rt_reflect_slider.curvalue = RT_Clamp(0, 10, Cvar_VariableValue("rt_reflect") * 10.0f);

	threads = Cvar_VariableIntValue("rt_threads");
	s_rt_threads_box.curvalue = RT_ThreadsIndex(threads);
	s_rt_weapons_box.curvalue = Cvar_VariableIntValue("rt_weapons") != 0;
}

static void RT_EnableFunc(void *unused)
{
	Cvar_SetValue("r_raytrace", (float)s_rt_enable_box.curvalue);
}

static void RT_ModeFunc(void *unused)
{
	Cvar_SetValue("rt_mode", (float)s_rt_mode_box.curvalue);
}

static void RT_QualityFunc(void *unused)
{
	int q = s_rt_quality_box.curvalue;
	if (q < 0) q = 0;
	if (q > 4) q = 4;
	Cvar_SetValue("rt_scale", (float)rt_quality_scales[q]);
}

static void RT_BouncesFunc(void *unused)
{
	Cvar_SetValue("rt_bounces", (float)s_rt_bounces_box.curvalue);
}

static void RT_AccumFunc(void *unused)
{
	Cvar_SetValue("rt_accum", (float)s_rt_accum_box.curvalue);
}

static void RT_ExposureFunc(void *unused)
{
	Cvar_SetValue("rt_exposure", s_rt_exposure_slider.curvalue / 10.0f);
}

static void RT_LightscaleFunc(void *unused)
{
	Cvar_SetValue("rt_lightscale", s_rt_lightscale_slider.curvalue / 10.0f);
}

static void RT_SkylightFunc(void *unused)
{
	Cvar_SetValue("rt_skylight", s_rt_skylight_slider.curvalue / 10.0f);
}

static void RT_ReflectFunc(void *unused)
{
	Cvar_SetValue("rt_reflect", s_rt_reflect_slider.curvalue / 10.0f);
}

static void RT_ThreadsFunc(void *unused)
{
	int i = s_rt_threads_box.curvalue;
	if (i < 0) i = 0;
	if (i > 4) i = 4;
	Cvar_SetValue("rt_threads", (float)rt_thread_counts[i]);
}

static void RT_WeaponsFunc(void *unused)
{
	Cvar_SetValue("rt_weapons", (float)s_rt_weapons_box.curvalue);
}

static void RT_DefaultsFunc(void *unused)
{
	Cvar_SetDefault("r_raytrace");
	Cvar_SetDefault("rt_mode");
	Cvar_SetDefault("rt_scale");
	Cvar_SetDefault("rt_bounces");
	Cvar_SetDefault("rt_accum");
	Cvar_SetDefault("rt_exposure");
	Cvar_SetDefault("rt_lightscale");
	Cvar_SetDefault("rt_skylight");
	Cvar_SetDefault("rt_reflect");
	Cvar_SetDefault("rt_threads");
	Cvar_SetDefault("rt_weapons");
	RT_SetMenuItemValues();
}

static void RT_MenuDraw(menuframework_s *self)
{
	M_Banner("m_banner_video");
	Menu_AdjustCursor(self, 1);
	Menu_Draw(self);
}

static void RT_MenuInit(void)
{
	static const char *yesno_names[] = {
		"no",
		"yes",
		0
	};
	static const char *mode_names[] = {
		"hybrid",
		"path trace",
		0
	};
	static const char *quality_names[] = {
		"fast",
		"low",
		"medium",
		"high",
		"ultra",
		0
	};
	static const char *bounce_names[] = {
		"off",
		"one bounce",
		0
	};
	static const char *thread_names[] = {
		"auto",
		"1",
		"2",
		"4",
		"8",
		0
	};
	int y;

	RT_EnsureCvars();

	memset(&s_rt_menu, 0, sizeof(s_rt_menu));
	s_rt_menu.x = viddef.width / 2;
	s_rt_menu.y = viddef.height / 2 - 110;
	s_rt_menu.nitems = 0;

	s_rt_enable_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_enable_box.generic.x = 0;
	s_rt_enable_box.generic.y = y = 0;
	s_rt_enable_box.generic.name = "raytracing";
	s_rt_enable_box.generic.callback = RT_EnableFunc;
	s_rt_enable_box.itemnames = yesno_names;
	s_rt_enable_box.generic.statusbar = "CPU BSP path tracer (Q2VKPT-style)";

	s_rt_mode_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_mode_box.generic.x = 0;
	s_rt_mode_box.generic.y = y += 10;
	s_rt_mode_box.generic.name = "lighting";
	s_rt_mode_box.generic.callback = RT_ModeFunc;
	s_rt_mode_box.itemnames = mode_names;
	s_rt_mode_box.generic.statusbar = "Hybrid uses baked lightmaps; path trace samples lights";

	s_rt_quality_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_quality_box.generic.x = 0;
	s_rt_quality_box.generic.y = y += 10;
	s_rt_quality_box.generic.name = "quality";
	s_rt_quality_box.generic.callback = RT_QualityFunc;
	s_rt_quality_box.itemnames = quality_names;
	s_rt_quality_box.generic.statusbar = "Internal resolution; higher is sharper and slower";

	s_rt_bounces_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_bounces_box.generic.x = 0;
	s_rt_bounces_box.generic.y = y += 10;
	s_rt_bounces_box.generic.name = "bounce light";
	s_rt_bounces_box.generic.callback = RT_BouncesFunc;
	s_rt_bounces_box.itemnames = bounce_names;
	s_rt_bounces_box.generic.statusbar = "Indirect bounce for color bleed and extra fill";

	s_rt_accum_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_accum_box.generic.x = 0;
	s_rt_accum_box.generic.y = y += 10;
	s_rt_accum_box.generic.name = "temporal filter";
	s_rt_accum_box.generic.callback = RT_AccumFunc;
	s_rt_accum_box.itemnames = yesno_names;
	s_rt_accum_box.generic.statusbar = "Accumulate samples while standing still";

	s_rt_exposure_slider.generic.type = MTYPE_SLIDER;
	s_rt_exposure_slider.generic.x = 0;
	s_rt_exposure_slider.generic.y = y += 20;
	s_rt_exposure_slider.generic.name = "exposure";
	s_rt_exposure_slider.generic.callback = RT_ExposureFunc;
	s_rt_exposure_slider.minvalue = 8;
	s_rt_exposure_slider.maxvalue = 50;
	s_rt_exposure_slider.generic.statusbar = "Overall brightness of the path tracer";

	s_rt_lightscale_slider.generic.type = MTYPE_SLIDER;
	s_rt_lightscale_slider.generic.x = 0;
	s_rt_lightscale_slider.generic.y = y += 10;
	s_rt_lightscale_slider.generic.name = "lightmaps";
	s_rt_lightscale_slider.generic.callback = RT_LightscaleFunc;
	s_rt_lightscale_slider.minvalue = 5;
	s_rt_lightscale_slider.maxvalue = 30;
	s_rt_lightscale_slider.generic.statusbar = "Baked lightmap intensity on floors and walls";

	s_rt_skylight_slider.generic.type = MTYPE_SLIDER;
	s_rt_skylight_slider.generic.x = 0;
	s_rt_skylight_slider.generic.y = y += 10;
	s_rt_skylight_slider.generic.name = "sky light";
	s_rt_skylight_slider.generic.callback = RT_SkylightFunc;
	s_rt_skylight_slider.minvalue = 0;
	s_rt_skylight_slider.maxvalue = 50;
	s_rt_skylight_slider.generic.statusbar = "Sun and window light on surfaces";

	s_rt_reflect_slider.generic.type = MTYPE_SLIDER;
	s_rt_reflect_slider.generic.x = 0;
	s_rt_reflect_slider.generic.y = y += 10;
	s_rt_reflect_slider.generic.name = "reflectiveness";
	s_rt_reflect_slider.generic.callback = RT_ReflectFunc;
	s_rt_reflect_slider.minvalue = 0;
	s_rt_reflect_slider.maxvalue = 10;
	s_rt_reflect_slider.generic.statusbar = "Glossy reflections on floors and walls";

	s_rt_threads_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_threads_box.generic.x = 0;
	s_rt_threads_box.generic.y = y += 20;
	s_rt_threads_box.generic.name = "threads";
	s_rt_threads_box.generic.callback = RT_ThreadsFunc;
	s_rt_threads_box.itemnames = thread_names;
	s_rt_threads_box.generic.statusbar = "Worker threads; auto uses all cores";

	s_rt_weapons_box.generic.type = MTYPE_SPINCONTROL;
	s_rt_weapons_box.generic.x = 0;
	s_rt_weapons_box.generic.y = y += 10;
	s_rt_weapons_box.generic.name = "view weapon";
	s_rt_weapons_box.generic.callback = RT_WeaponsFunc;
	s_rt_weapons_box.itemnames = yesno_names;
	s_rt_weapons_box.generic.statusbar = "Draw the first-person gun over the trace";

	s_rt_defaults_action.generic.type = MTYPE_ACTION;
	s_rt_defaults_action.generic.x = 0;
	s_rt_defaults_action.generic.y = y += 20;
	s_rt_defaults_action.generic.name = "reset defaults";
	s_rt_defaults_action.generic.callback = RT_DefaultsFunc;
	s_rt_defaults_action.generic.statusbar = "Restore default raytracing settings";

	RT_SetMenuItemValues();

	s_rt_menu.draw = RT_MenuDraw;
	s_rt_menu.key = NULL;

	Menu_AddItem(&s_rt_menu, (void *)&s_rt_enable_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_mode_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_quality_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_bounces_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_accum_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_exposure_slider);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_lightscale_slider);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_skylight_slider);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_reflect_slider);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_threads_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_weapons_box);
	Menu_AddItem(&s_rt_menu, (void *)&s_rt_defaults_action);
}

void M_Menu_Raytrace_f(void)
{
	RT_MenuInit();
	M_PushMenu(&s_rt_menu);
}
