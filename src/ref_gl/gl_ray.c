/*
 * Path tracer for this OpenGL 1.x Quake II client.
 *
 * On macOS, rt_gpu 1 (default) runs fullscreen path-trace on Metal.
 * CPU BSP tracing remains as fallback (rt_gpu 0 or Metal init failure).
 *
 * rt_mode:
 *   0 = lightmaps   — raster world + progressive RT luxel bake
 *   1 = raytracing  — hybrid raster + soft RT shadows + reflection overlay
 *   2 = path trace  — fullscreen path-tracer blit
 *
 * Console:
 *   r_raytrace 1     enable (default on)
 *   rt_gpu 1         Metal GPU tracer (macOS)
 *   rt_scale 2       bake budget / path-trace resolution divisor
 *   rt_mode 0|1|2    lighting mode (see above)
 *   rt_bounces 1     0 or 1 extra bounce
 *   rt_accum 1       temporal filter (path trace)
 */
#include "gl_local.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include "gl_ray_gpu.h"
#endif

#define RT_MAX_DIST		16384.0f
#define RT_EPS			0.125f
#define RT_MAX_LIGHTS	384
#define RT_MAX_THREADS	16
#define RT_STACK		72
#define RT_MAX_ENTS		128
#define RT_LM_BUDGET_BASE	256

typedef struct rt_ent_s {
	vec3_t		origin;
	vec3_t		axis[3];
	vec3_t		mins;
	vec3_t		maxs;
	vec3_t		albedo;
	qboolean	valid;
} rt_ent_t;

typedef struct {
	qboolean	hit;
	qboolean	sky;
	qboolean	is_entity;
	float		t;
	vec3_t		pos;
	vec3_t		normal;
	msurface_t	*surf;
	const rt_ent_t	*ent;
} rt_hit_t;

typedef struct {
	vec3_t		origin;
	vec3_t		dir;
} rt_ray_t;

typedef struct {
	msurface_t	*surf;
	vec3_t		center;
	vec3_t		emit;
	float		area;
	float		weight;
} rt_light_t;

typedef struct {
	int			y0, y1;
	unsigned	seed;
	int			mode;		/* 0 = full primary, 1 = reflection only */
} rt_job_t;

static cvar_t	*r_raytrace;
static cvar_t	*rt_scale;
static cvar_t	*rt_mode;
static cvar_t	*rt_bounces;
static cvar_t	*rt_accum;
static cvar_t	*rt_exposure;
static cvar_t	*rt_lightscale;
static cvar_t	*rt_skylight;
static cvar_t	*rt_reflect;
static cvar_t	*rt_threads;
static cvar_t	*rt_weapons;
static cvar_t	*rt_dump;
static cvar_t	*rt_gpu;
static cvar_t	*rt_spp;
static cvar_t	*rt_log;
static FILE		*rt_runlog;
static int		rt_runlog_frame;

static byte		*rt_pixels;
static float	*rt_frame_buf;
static float	*rt_accum_buf;
static float	*rt_zeye;
static int		rt_buf_w, rt_buf_h, rt_buf_size;

extern float	skybox_farz;
static GLuint	rt_texnum;
static qboolean	rt_tex_ready;
static GLuint	rt_refl_texnum;
static qboolean	rt_refl_tex_ready;
static byte		*rt_refl_pixels;

static rt_light_t	rt_lights[RT_MAX_LIGHTS];
static int			rt_num_lights;
static float		rt_light_sum;
static rt_light_t	rt_skylights[RT_MAX_LIGHTS];
static int			rt_num_skylights;
static const bspModel_t	*rt_light_world;

static rt_ent_t		rt_ents[RT_MAX_ENTS];
static int			rt_num_ents;
static int			rt_lm_cursor;
static int			rt_lm_surf_cursor;
static int			rt_lm_algo;

static vec3_t	rt_prev_org;
static vec3_t	rt_prev_fwd;
static int		rt_first_frame = 1;

static float	rt_tanx, rt_tany;
static int		rt_w, rt_h;
static unsigned	rt_frame_seed;
static int		rt_stat_hit, rt_stat_sky, rt_stat_miss;

static void RT_RunLog(const char *fmt, ...)
{
	va_list argptr;
	char msg[1024];
	time_t now;
	struct tm *tm;

	if (!rt_log || !rt_log->integer)
		return;

	va_start(argptr, fmt);
	vsnprintf(msg, sizeof(msg), fmt, argptr);
	va_end(argptr);

	Com_Printf("%s", msg);

	if (!rt_runlog) {
		/* Project-local run log so the agent can track sessions */
		rt_runlog = fopen("/Users/h8/_dev/quake2/logs/rt-latest.log", "a");
		if (rt_runlog) {
			time(&now);
			tm = localtime(&now);
			fprintf(rt_runlog, "\n======== RT session %04d-%02d-%02d %02d:%02d:%02d ========\n",
				tm ? tm->tm_year + 1900 : 0,
				tm ? tm->tm_mon + 1 : 0,
				tm ? tm->tm_mday : 0,
				tm ? tm->tm_hour : 0,
				tm ? tm->tm_min : 0,
				tm ? tm->tm_sec : 0);
			fflush(rt_runlog);
		}
	}
	if (rt_runlog) {
		fputs(msg, rt_runlog);
		fflush(rt_runlog);
	}
}

static unsigned RT_Rand(unsigned *s)
{
	unsigned x = *s;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*s = x ? x : 0xA341316Cu;
	return x;
}

static float RT_Frand(unsigned *s)
{
	return (RT_Rand(s) & 0xFFFFFF) * (1.0f / 16777216.0f);
}

static void RT_SunDir(vec3_t dir)
{
	dir[0] = 0.52f;
	dir[1] = 0.28f;
	dir[2] = 0.81f;
	VectorNormalize(dir);
}

static void RT_SkyColor(const vec3_t dir, vec3_t out)
{
	float sun;
	vec3_t sunDir, tex;

	/* Actual skybox (unit1 red, etc) so GI matches what you see through roof holes */
	R_SampleSkyColor(dir, tex);
	out[0] = tex[0] * 1.35f + 0.08f;
	out[1] = tex[1] * 1.25f + 0.06f;
	out[2] = tex[2] * 1.15f + 0.05f;

	RT_SunDir(sunDir);
	sun = DotProduct(dir, sunDir);
	if (sun > 0.0f) {
		sun = sun * sun;
		sun = sun * sun;
		sun = sun * sun;
		out[0] += sun * 4.5f;
		out[1] += sun * 3.6f;
		out[2] += sun * 2.4f;
	}
}

static qboolean RT_RayAABB(const vec3_t o, const vec3_t d, float tmin, float tmax, const float *mins, const float *maxs)
{
	int i;

	for (i = 0; i < 3; i++) {
		if (d[i] > -1e-8f && d[i] < 1e-8f) {
			if (o[i] < mins[i] || o[i] > maxs[i])
				return false;
		} else {
			float inv = 1.0f / d[i];
			float t0 = (mins[i] - o[i]) * inv;
			float t1 = (maxs[i] - o[i]) * inv;
			if (t0 > t1) {
				float tmp = t0;
				t0 = t1;
				t1 = tmp;
			}
			if (t0 > tmin) tmin = t0;
			if (t1 < tmax) tmax = t1;
			if (tmax < tmin)
				return false;
		}
	}
	return true;
}

static qboolean RT_PointInPoly(const msurface_t *surf, const vec3_t p)
{
	const glpoly_t *poly;
	const float *verts, *a, *b, *c;
	vec3_t ab, bc, ca, ap, bp, cp, n, cr;
	int i, nv;
	float area2, tola, u, v, w;

	poly = surf->polys;
	if (!poly || poly->numverts < 3)
		return false;

	verts = &poly->verts[0][0];
	nv = poly->numverts;
	a = verts;
	for (i = 1; i < nv - 1; i++) {
		b = verts + i * VERTEXSIZE;
		c = verts + (i + 1) * VERTEXSIZE;
		VectorSubtract(b, a, ab);
		VectorSubtract(c, a, ca);
		CrossProduct(ab, ca, n);
		area2 = DotProduct(n, n);
		if (area2 < 1.0f)
			continue;
		tola = area2 * 0.05f;
		VectorSubtract(c, b, bc);
		VectorSubtract(a, c, ca);
		VectorSubtract(p, a, ap);
		VectorSubtract(p, b, bp);
		VectorSubtract(p, c, cp);
		CrossProduct(ab, ap, cr);
		u = DotProduct(cr, n);
		CrossProduct(bc, bp, cr);
		v = DotProduct(cr, n);
		CrossProduct(ca, cp, cr);
		w = DotProduct(cr, n);
		if ((u >= -tola && v >= -tola && w >= -tola) ||
			(u <= tola && v <= tola && w <= tola))
			return true;
	}
	return false;
}

static void RT_SurfNormal(const msurface_t *surf, vec3_t n)
{
	VectorCopy(surf->plane->normal, n);
	if (surf->flags & SURF_PLANEBACK)
		VectorNegate(n, n);
}

static qboolean RT_IntersectSurf(const msurface_t *surf, const vec3_t o, const vec3_t d, float tmin, float tmax, float *t_out, vec3_t pos, vec3_t n)
{
	float dist, denom, t;
	vec3_t p;

	if (!surf->texinfo || (surf->texinfo->flags & SURF_NODRAW))
		return false;
	if (!surf->polys)
		return false;

	RT_SurfNormal(surf, n);
	denom = DotProduct(d, n);
	if (denom > -1e-5f)
		return false;

	if (surf->flags & SURF_PLANEBACK)
		dist = DotProduct(o, n) + surf->plane->dist;
	else
		dist = DotProduct(o, n) - surf->plane->dist;
	t = -dist / denom;
	if (t < tmin || t > tmax)
		return false;

	VectorMA(o, t, d, p);
	if (!RT_PointInPoly(surf, p))
		return false;

	*t_out = t;
	VectorCopy(p, pos);
	return true;
}

static void RT_TestLeaf(const mleaf_t *leaf, const vec3_t o, const vec3_t d, float tmin, float tmax, rt_hit_t *hit)
{
	msurface_t **mark;
	int c;
	float t;
	vec3_t pos, n;

	mark = leaf->firstmarksurface;
	c = leaf->nummarksurfaces;
	while (c-- > 0) {
		msurface_t *surf = *mark++;
		if (RT_IntersectSurf(surf, o, d, tmin, hit->hit ? hit->t : tmax, &t, pos, n)) {
			hit->hit = true;
			hit->t = t;
			hit->surf = surf;
			VectorCopy(pos, hit->pos);
			VectorCopy(n, hit->normal);
			hit->sky = (surf->texinfo->flags & SURF_SKY) ? true : false;
			tmax = t;
		}
	}
}

static void RT_Traverse(const mnode_t *node, const vec3_t o, const vec3_t d, float tmin, float tmax, rt_hit_t *hit)
{
	const mnode_t *stack[RT_STACK];
	int sp = 0;

	(void)tmin;
	stack[sp++] = node;

	while (sp > 0) {
		const mnode_t *n;
		const cplane_t *plane;
		float dist, denom;
		int near_child, i;
		msurface_t *surf;
		float t, nearest;
		vec3_t pos, nm;

		n = stack[--sp];
		nearest = hit->hit ? hit->t : tmax;
		if (!RT_RayAABB(o, d, 0.0f, nearest, n->minmaxs, n->minmaxs + 3))
			continue;

		if (n->contents != CONTENTS_NODE) {
			if (n->contents != CONTENTS_SOLID)
				RT_TestLeaf((const mleaf_t *)n, o, d, 0.0f, nearest, hit);
			continue;
		}

		surf = r_worldmodel->surfaces + n->firstsurface;
		for (i = 0; i < n->numsurfaces; i++, surf++) {
			if (RT_IntersectSurf(surf, o, d, 0.0f, nearest, &t, pos, nm)) {
				hit->hit = true;
				hit->t = t;
				hit->surf = surf;
				VectorCopy(pos, hit->pos);
				VectorCopy(nm, hit->normal);
				hit->sky = (surf->texinfo && (surf->texinfo->flags & SURF_SKY)) ? true : false;
				nearest = t;
			}
		}

		plane = n->plane;
		if (plane->type < 3) {
			dist = o[plane->type] - plane->dist;
			denom = d[plane->type];
		} else {
			dist = DotProduct(o, plane->normal) - plane->dist;
			denom = DotProduct(d, plane->normal);
		}
		near_child = (dist >= 0.0f) ? 0 : 1;
		(void)denom;

		if (sp < RT_STACK - 1) {
			stack[sp++] = n->children[near_child ^ 1];
			stack[sp++] = n->children[near_child];
		}
	}
}

static qboolean RT_RayOrientedAABB(const vec3_t o, const vec3_t d, float tmin, float tmax,
	const rt_ent_t *e, float *t_out, vec3_t pos, vec3_t n)
{
	vec3_t localO, localD, hitLocal, absD;
	float t0 = tmin, t1 = tmax;
	int i, axis = -1;

	for (i = 0; i < 3; i++) {
		localO[i] = (o[0] - e->origin[0]) * e->axis[i][0]
			+ (o[1] - e->origin[1]) * e->axis[i][1]
			+ (o[2] - e->origin[2]) * e->axis[i][2];
		localD[i] = d[0] * e->axis[i][0] + d[1] * e->axis[i][1] + d[2] * e->axis[i][2];
		absD[i] = fabsf(localD[i]);
	}

	for (i = 0; i < 3; i++) {
		if (absD[i] < 1e-8f) {
			if (localO[i] < e->mins[i] || localO[i] > e->maxs[i])
				return false;
		} else {
			float inv = 1.0f / localD[i];
			float nearT = (e->mins[i] - localO[i]) * inv;
			float farT = (e->maxs[i] - localO[i]) * inv;
			int hitAxis = i;
			if (nearT > farT) {
				float tmp = nearT;
				nearT = farT;
				farT = tmp;
			}
			if (nearT > t0) {
				t0 = nearT;
				axis = hitAxis;
			}
			if (farT < t1)
				t1 = farT;
			if (t0 > t1)
				return false;
		}
	}

	if (t0 < tmin || t0 > tmax)
		return false;

	VectorMA(o, t0, d, pos);
	for (i = 0; i < 3; i++)
		hitLocal[i] = localO[i] + localD[i] * t0;

	VectorClear(n);
	if (axis < 0) {
		float best = -1.0f;
		for (i = 0; i < 3; i++) {
			float a = fabsf(hitLocal[i] - e->mins[i]);
			float b = fabsf(hitLocal[i] - e->maxs[i]);
			if (a < b && (best < 0.0f || a < best)) {
				best = a;
				axis = i;
			} else if (best < 0.0f || b < best) {
				best = b;
				axis = i;
			}
		}
	}
	if (axis < 0)
		axis = 2;
	{
		float sign = (hitLocal[axis] > 0.5f * (e->mins[axis] + e->maxs[axis])) ? 1.0f : -1.0f;
		n[0] = e->axis[axis][0] * sign;
		n[1] = e->axis[axis][1] * sign;
		n[2] = e->axis[axis][2] * sign;
	}
	*t_out = t0;
	return true;
}

static void RT_TraceEntities(const vec3_t o, const vec3_t d, float tmin, float tmax, rt_hit_t *hit)
{
	int i;

	for (i = 0; i < rt_num_ents; i++) {
		const rt_ent_t *e = &rt_ents[i];
		float t;
		vec3_t pos, n;

		if (!e->valid)
			continue;
		if (!RT_RayOrientedAABB(o, d, tmin, hit->hit ? hit->t : tmax, e, &t, pos, n))
			continue;
		hit->hit = true;
		hit->sky = false;
		hit->is_entity = true;
		hit->t = t;
		VectorCopy(pos, hit->pos);
		VectorCopy(n, hit->normal);
		hit->surf = NULL;
		hit->ent = e;
	}
}

static qboolean RT_Trace(const vec3_t o, const vec3_t d, float tmin, float tmax, rt_hit_t *hit)
{
	memset(hit, 0, sizeof(*hit));
	hit->t = tmax;
	if (r_worldmodel)
		RT_Traverse(r_worldmodel->nodes, o, d, tmin, tmax, hit);
	RT_TraceEntities(o, d, tmin, tmax, hit);
	return hit->hit;
}

static qboolean RT_Occluded(const vec3_t o, const vec3_t d, float tmax)
{
	rt_hit_t hit;
	return RT_Trace(o, d, RT_EPS, tmax - RT_EPS * 2.0f, &hit) && !hit.sky;
}

static void RT_AvgSkinAlbedo(image_t *image, vec3_t out)
{
	int i, n;
	double r = 0, g = 0, b = 0;
	const byte *p;

	VectorSet(out, 0.55f, 0.45f, 0.35f);
	if (!image || !image->pixels || image->width < 1 || image->height < 1)
		return;
	n = image->width * image->height;
	if (n > 4096)
		n = 4096;
	p = image->pixels;
	for (i = 0; i < n; i++, p += 3) {
		r += p[0];
		g += p[1];
		b += p[2];
	}
	out[0] = (float)(r / (n * 255.0));
	out[1] = (float)(g / (n * 255.0));
	out[2] = (float)(b / (n * 255.0));
}

static void RT_CollectEntities(void)
{
	entity_t *ent, *lastE;

	rt_num_ents = 0;
	if (!r_newrefdef.num_entities)
		return;

	lastE = r_newrefdef.entities + r_newrefdef.num_entities;
	for (ent = r_newrefdef.entities; ent != lastE && rt_num_ents < RT_MAX_ENTS; ent++) {
		model_t *model;
		aliasModel_t *alias;
		aliasFrame_t *frame;
		rt_ent_t *e;
		image_t *skin;
		int f;

		if (ent->flags & RF_BEAM)
			continue;
		model = ent->model;
		if (!model || model->type != mod_alias || !model->aliasModel)
			continue;
		alias = model->aliasModel;
		if (!alias->numFrames || !alias->frames)
			continue;

		f = ent->frame % alias->numFrames;
		if (f < 0)
			f = 0;
		frame = &alias->frames[f];

		e = &rt_ents[rt_num_ents];
		memset(e, 0, sizeof(*e));
		VectorCopy(ent->origin, e->origin);
		VectorCopy(ent->axis[0], e->axis[0]);
		VectorCopy(ent->axis[1], e->axis[1]);
		VectorCopy(ent->axis[2], e->axis[2]);
		VectorCopy(frame->mins, e->mins);
		VectorCopy(frame->maxs, e->maxs);
		e->mins[0] -= 1.0f; e->mins[1] -= 1.0f; e->mins[2] -= 1.0f;
		e->maxs[0] += 1.0f; e->maxs[1] += 1.0f; e->maxs[2] += 1.0f;

		skin = ent->skin;
		if (!skin && alias->numMeshes && alias->meshes[0].numSkins)
			skin = alias->meshes[0].skins[0].image;
		RT_AvgSkinAlbedo(skin, e->albedo);
		if (ent->flags & RF_SHELL_RED) {
			e->albedo[0] = 1.0f; e->albedo[1] = 0.2f; e->albedo[2] = 0.2f;
		} else if (ent->flags & RF_SHELL_GREEN) {
			e->albedo[0] = 0.2f; e->albedo[1] = 1.0f; e->albedo[2] = 0.2f;
		} else if (ent->flags & RF_SHELL_BLUE) {
			e->albedo[0] = 0.2f; e->albedo[1] = 0.2f; e->albedo[2] = 1.0f;
		}
		e->valid = true;
		rt_num_ents++;
	}
}

static void RT_SampleTexture(const msurface_t *surf, const vec3_t p, vec3_t albedo)
{
	mtexinfo_t *tex;
	image_t *image;
	float s, t;
	int x, y, w, h;
	const byte *pix;

	VectorSet(albedo, 0.65f, 0.65f, 0.65f);
	tex = surf->texinfo;
	if (!tex)
		return;

	image = tex->image;
	if (tex->next && tex->numframes > 1) {
		int c = ((int)(r_newrefdef.time * 2.0f)) % tex->numframes;
		mtexinfo_t *step = tex;
		while (c-- > 0 && step->next)
			step = step->next;
		image = step->image;
	}
	if (!image || !image->pixels || image->width < 1 || image->height < 1)
		return;

	s = DotProduct(p, tex->vecs[0]) + tex->vecs[0][3];
	t = DotProduct(p, tex->vecs[1]) + tex->vecs[1][3];
	if (tex->flags & SURF_FLOWING)
		s -= r_newrefdef.time * 40.0f;

	w = image->width;
	h = image->height;
	x = (int)floorf(s) % w;
	y = (int)floorf(t) % h;
	if (x < 0) x += w;
	if (y < 0) y += h;

	pix = image->pixels + (y * w + x) * 3;
	albedo[0] = pix[0] * ONEDIV255;
	albedo[1] = pix[1] * ONEDIV255;
	albedo[2] = pix[2] * ONEDIV255;
}

static void RT_SampleLightmap(const msurface_t *surf, const vec3_t p, vec3_t lm)
{
	mtexinfo_t *tex;
	float s, t, ls, lt, fracs, fract;
	int smax, tmax, x0, y0, x1, y1, maps, idx00, idx10, idx01, idx11;
	byte *base;

	VectorSet(lm, 0.22f, 0.22f, 0.24f);
	if (!surf->samples || !surf->texinfo)
		return;
	if (surf->texinfo->flags & (SURF_SKY | SURF_WARP | SURF_TRANS33 | SURF_TRANS66))
		return;

	tex = surf->texinfo;
	s = DotProduct(p, tex->vecs[0]) + tex->vecs[0][3] - surf->texturemins[0];
	t = DotProduct(p, tex->vecs[1]) + tex->vecs[1][3] - surf->texturemins[1];
	smax = (surf->extents[0] >> 4) + 1;
	tmax = (surf->extents[1] >> 4) + 1;
	if (smax < 1 || tmax < 1)
		return;

	ls = s * (1.0f / 16.0f);
	lt = t * (1.0f / 16.0f);
	x0 = (int)floorf(ls);
	y0 = (int)floorf(lt);
	fracs = ls - x0;
	fract = lt - y0;
	if (x0 < 0) { x0 = 0; fracs = 0; }
	if (y0 < 0) { y0 = 0; fract = 0; }
	if (x0 >= smax) { x0 = smax - 1; fracs = 0; }
	if (y0 >= tmax) { y0 = tmax - 1; fract = 0; }
	x1 = x0 + 1; if (x1 >= smax) x1 = smax - 1;
	y1 = y0 + 1; if (y1 >= tmax) y1 = tmax - 1;

	VectorClear(lm);
	base = surf->samples;
	for (maps = 0; maps < MAXLIGHTMAPS && surf->styles[maps] != 255; maps++) {
		float sr = r_newrefdef.lightstyles[surf->styles[maps]].rgb[0];
		float sg = r_newrefdef.lightstyles[surf->styles[maps]].rgb[1];
		float sb = r_newrefdef.lightstyles[surf->styles[maps]].rgb[2];
		byte *blk = base + maps * smax * tmax * 3;
		idx00 = (y0 * smax + x0) * 3;
		idx10 = (y0 * smax + x1) * 3;
		idx01 = (y1 * smax + x0) * 3;
		idx11 = (y1 * smax + x1) * 3;
		lm[0] += ((blk[idx00] * (1 - fracs) + blk[idx10] * fracs) * (1 - fract)
			+ (blk[idx01] * (1 - fracs) + blk[idx11] * fracs) * fract) * sr;
		lm[1] += ((blk[idx00 + 1] * (1 - fracs) + blk[idx10 + 1] * fracs) * (1 - fract)
			+ (blk[idx01 + 1] * (1 - fracs) + blk[idx11 + 1] * fracs) * fract) * sg;
		lm[2] += ((blk[idx00 + 2] * (1 - fracs) + blk[idx10 + 2] * fracs) * (1 - fract)
			+ (blk[idx01 + 2] * (1 - fracs) + blk[idx11 + 2] * fracs) * fract) * sb;
	}
	{
		float scale = gl_modulate->value * ONEDIV255 * 1.55f;
		if (rt_lightscale && rt_lightscale->value > 0.1f)
			scale *= rt_lightscale->value;
		VectorScale(lm, scale, lm);
	}
}

static void RT_CosineDir(const vec3_t n, unsigned *rng, vec3_t out)
{
	vec3_t t, b;
	float u, v, r, phi, x, y, z;

	if (fabsf(n[2]) < 0.9f) {
		t[0] = n[1]; t[1] = -n[0]; t[2] = 0.0f;
	} else {
		t[0] = 0.0f; t[1] = n[2]; t[2] = -n[1];
	}
	VectorNormalize(t);
	CrossProduct(n, t, b);
	u = RT_Frand(rng);
	v = RT_Frand(rng);
	r = sqrtf(u);
	phi = v * 6.28318530718f;
	x = r * cosf(phi);
	y = r * sinf(phi);
	z = sqrtf(1.0f - u);
	out[0] = t[0] * x + b[0] * y + n[0] * z;
	out[1] = t[1] * x + b[1] * y + n[1] * z;
	out[2] = t[2] * x + b[2] * y + n[2] * z;
}

static void RT_PolyStats(const msurface_t *surf, vec3_t center, float *area)
{
	const glpoly_t *poly = surf->polys;
	const float *v0, *v1, *v2;
	vec3_t acc, e1, e2, cr;
	int i, n;

	VectorClear(acc);
	*area = 0.0f;
	if (!poly || poly->numverts < 3)
		return;

	n = poly->numverts;
	v0 = &poly->verts[0][0];
	for (i = 0; i < n; i++) {
		const float *v = v0 + i * VERTEXSIZE;
		VectorAdd(acc, v, acc);
	}
	VectorScale(acc, 1.0f / (float)n, center);

	for (i = 1; i < n - 1; i++) {
		v1 = v0 + i * VERTEXSIZE;
		v2 = v0 + (i + 1) * VERTEXSIZE;
		VectorSubtract(v1, v0, e1);
		VectorSubtract(v2, v0, e2);
		CrossProduct(e1, e2, cr);
		*area += 0.5f * VectorLength(cr);
	}
}

static void RT_RandomPolyPoint(const msurface_t *surf, unsigned *rng, vec3_t out)
{
	const glpoly_t *poly = surf->polys;
	const float *v0, *v1, *v2;
	int n, tri;
	float u, v, w;

	if (!poly || poly->numverts < 3) {
		VectorClear(out);
		return;
	}
	n = poly->numverts;
	tri = (int)(RT_Frand(rng) * (float)(n - 2));
	if (tri < 0)
		tri = 0;
	if (tri > n - 3)
		tri = n - 3;
	v0 = &poly->verts[0][0];
	v1 = v0 + (tri + 1) * VERTEXSIZE;
	v2 = v0 + (tri + 2) * VERTEXSIZE;
	u = RT_Frand(rng);
	v = RT_Frand(rng);
	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}
	w = 1.0f - u - v;
	out[0] = v0[0] * w + v1[0] * u + v2[0] * v;
	out[1] = v0[1] * w + v1[1] * u + v2[1] * v;
	out[2] = v0[2] * w + v1[2] * u + v2[2] * v;
}

static void RT_CollectLights(void)
{
	int i;
	vec3_t albedo;

	rt_num_lights = 0;
	rt_num_skylights = 0;
	rt_light_sum = 0.0f;
	rt_light_world = r_worldmodel;
	if (!r_worldmodel)
		return;

	for (i = 0; i < r_worldmodel->numsurfaces; i++) {
		msurface_t *s = r_worldmodel->surfaces + i;
		rt_light_t *L;
		float lum, intensity, area;
		vec3_t center;

		if (!s->texinfo || !s->polys)
			continue;

		if (s->texinfo->flags & SURF_SKY) {
			if (rt_num_skylights >= RT_MAX_LIGHTS)
				continue;
			RT_PolyStats(s, center, &area);
			if (area < 8.0f)
				continue;
			L = &rt_skylights[rt_num_skylights++];
			L->surf = s;
			VectorCopy(center, L->center);
			L->area = area;
			L->emit[0] = 4.5f;
			L->emit[1] = 4.1f;
			L->emit[2] = 3.4f;
			L->weight = area;
			continue;
		}

		if (!(s->texinfo->flags & SURF_LIGHT))
			continue;
		if (rt_num_lights >= RT_MAX_LIGHTS)
			continue;

		L = &rt_lights[rt_num_lights];
		L->surf = s;
		RT_PolyStats(s, L->center, &L->area);
		if (L->area < 1.0f)
			continue;

		RT_SampleTexture(s, L->center, albedo);
		intensity = s->texinfo->value * 0.012f;
		if (intensity < 0.8f)
			intensity = 0.8f;
		VectorScale(albedo, intensity, L->emit);
		lum = L->emit[0] * 0.3f + L->emit[1] * 0.6f + L->emit[2] * 0.1f;
		L->weight = lum * L->area;
		if (L->weight < 0.01f)
			L->weight = 0.01f;
		rt_light_sum += L->weight;
		rt_num_lights++;
	}
}

static void RT_Dlights(const rt_hit_t *hit, vec3_t out)
{
	int i;
	dlight_t *dl;
	vec3_t toL, origin;
	float dist, ndotl, att, tmax;

	VectorClear(out);
	VectorMA(hit->pos, RT_EPS, hit->normal, origin);

	for (i = 0, dl = r_newrefdef.dlights; i < r_newrefdef.num_dlights; i++, dl++) {
		VectorSubtract(dl->origin, hit->pos, toL);
		dist = VectorNormalize(toL);
		if (dist < 1.0f || dist > dl->intensity)
			continue;
		ndotl = DotProduct(hit->normal, toL);
		if (ndotl <= 0.0f)
			continue;
		tmax = dist;
		if (RT_Occluded(origin, toL, tmax))
			continue;
		att = 1.0f - dist / dl->intensity;
		att = att * att * ndotl;
		VectorMA(out, att, dl->color, out);
	}
}

static void RT_NEE(const rt_hit_t *hit, unsigned *rng, vec3_t out)
{
	rt_light_t *L;
	vec3_t toL, origin, ln;
	float dist, ndotl, ldot, pdf, pick, w, att;
	int i;

	VectorClear(out);
	if (rt_num_lights < 1 || rt_light_sum <= 0.0f)
		return;

	pick = RT_Frand(rng) * rt_light_sum;
	w = 0.0f;
	L = &rt_lights[0];
	for (i = 0; i < rt_num_lights; i++) {
		w += rt_lights[i].weight;
		if (pick <= w) {
			L = &rt_lights[i];
			break;
		}
	}

	if (L->surf == hit->surf)
		return;

	VectorMA(hit->pos, RT_EPS, hit->normal, origin);
	VectorSubtract(L->center, hit->pos, toL);
	dist = VectorNormalize(toL);
	if (dist < 2.0f)
		return;
	ndotl = DotProduct(hit->normal, toL);
	if (ndotl <= 0.0f)
		return;

	RT_SurfNormal(L->surf, ln);
	ldot = -DotProduct(ln, toL);
	if (ldot < 0.0f)
		ldot = 0.0f;

	if (RT_Occluded(origin, toL, dist))
		return;

	pdf = L->weight / rt_light_sum;
	if (pdf < 1e-6f)
		return;
	att = (ndotl * ldot * L->area) / (dist * dist * pdf);
	att *= 0.55f;
	out[0] = L->emit[0] * att;
	out[1] = L->emit[1] * att;
	out[2] = L->emit[2] * att;
}

static void RT_SkyNEE(const rt_hit_t *hit, unsigned *rng, vec3_t out)
{
	vec3_t origin, dir, sky, target, toL, ln;
	rt_hit_t h;
	float ndotl, dist, gain, ldot;
	rt_light_t *L;

	VectorClear(out);
	gain = (rt_skylight && rt_skylight->value > 0.0f) ? rt_skylight->value : 0.0f;
	if (gain <= 0.01f)
		return;

	VectorMA(hit->pos, RT_EPS * 2.0f, hit->normal, origin);

	RT_SunDir(dir);
	dir[0] += (RT_Frand(rng) - 0.5f) * 0.06f;
	dir[1] += (RT_Frand(rng) - 0.5f) * 0.06f;
	VectorNormalize(dir);
	ndotl = DotProduct(hit->normal, dir);
	if (ndotl > 0.02f) {
		if (!RT_Trace(origin, dir, 0.0f, RT_MAX_DIST, &h) || h.sky) {
			RT_SkyColor(dir, sky);
			VectorMA(out, ndotl * gain * 2.2f, sky, out);
		}
	}

	if (rt_num_skylights > 0) {
		L = &rt_skylights[(int)(RT_Frand(rng) * (float)rt_num_skylights) % rt_num_skylights];
		RT_RandomPolyPoint(L->surf, rng, target);
		VectorSubtract(target, hit->pos, toL);
		dist = VectorNormalize(toL);
		ndotl = DotProduct(hit->normal, toL);
		if (dist > 4.0f && ndotl > 0.02f) {
			RT_SurfNormal(L->surf, ln);
			ldot = -DotProduct(ln, toL);
			if (ldot < 0.05f)
				ldot = 0.05f;
			if (!RT_Trace(origin, toL, 0.0f, dist + 8.0f, &h) || h.sky) {
				RT_SkyColor(toL, sky);
				VectorMA(out, ndotl * ldot * gain * 1.4f, sky, out);
			}
		}
	}
}

static void RT_Shade(const rt_ray_t *ray, const rt_hit_t *hit, unsigned *rng, int bounce, vec3_t color);

static void RT_ReflectDir(const vec3_t in, const vec3_t n, unsigned *rng, float rough, vec3_t out)
{
	vec3_t r, t, b;
	float ndoti, x, y;

	ndoti = DotProduct(in, n);
	VectorMA(in, -2.0f * ndoti, n, r);
	VectorNormalize(r);
	if (rough < 0.01f) {
		VectorCopy(r, out);
		return;
	}
	if (fabsf(r[2]) < 0.9f) {
		t[0] = r[1]; t[1] = -r[0]; t[2] = 0.0f;
	} else {
		t[0] = 0.0f; t[1] = r[2]; t[2] = -r[1];
	}
	VectorNormalize(t);
	CrossProduct(r, t, b);
	x = (RT_Frand(rng) * 2.0f - 1.0f) * rough;
	y = (RT_Frand(rng) * 2.0f - 1.0f) * rough;
	out[0] = r[0] + t[0] * x + b[0] * y;
	out[1] = r[1] + t[1] * x + b[1] * y;
	out[2] = r[2] + t[2] * x + b[2] * y;
	VectorNormalize(out);
	if (DotProduct(out, n) < 0.02f)
		VectorCopy(r, out);
}

static void RT_Glossy(const rt_ray_t *ray, const rt_hit_t *hit, unsigned *rng, vec3_t spec)
{
	rt_ray_t r2;
	rt_hit_t h2;

	VectorClear(spec);
	VectorMA(hit->pos, RT_EPS * 2.0f, hit->normal, r2.origin);
	RT_ReflectDir(ray->dir, hit->normal, rng, 0.16f, r2.dir);
	if (DotProduct(r2.dir, hit->normal) < 0.01f)
		return;
	if (!RT_Trace(r2.origin, r2.dir, 0.0f, RT_MAX_DIST, &h2) || h2.sky)
		RT_SkyColor(r2.dir, spec);
	else
		RT_Shade(&r2, &h2, rng, 0, spec);
}

static void RT_Bounce(const rt_hit_t *hit, unsigned *rng, int bounce, vec3_t gi)
{
	/* bounce may land on brush or entity */
	rt_ray_t ray2;
	rt_hit_t hit2;
	vec3_t col;

	VectorClear(gi);
	if (bounce <= 0)
		return;

	VectorMA(hit->pos, RT_EPS, hit->normal, ray2.origin);
	RT_CosineDir(hit->normal, rng, ray2.dir);
	if (!RT_Trace(ray2.origin, ray2.dir, 0.0f, RT_MAX_DIST, &hit2)) {
		RT_SkyColor(ray2.dir, gi);
		return;
	}
	if (hit2.sky) {
		RT_SkyColor(ray2.dir, gi);
		return;
	}
	RT_Shade(&ray2, &hit2, rng, bounce - 1, col);
	VectorCopy(col, gi);
}

static void RT_ShadeEntity(const rt_ray_t *ray, const rt_hit_t *hit, unsigned *rng, int bounce, vec3_t color)
{
	vec3_t lighting, dl, nee, gi;
	const rt_ent_t *e = hit->ent;

	VectorClear(lighting);
	RT_Dlights(hit, dl);
	VectorAdd(lighting, dl, lighting);
	RT_NEE(hit, rng, nee);
	VectorMA(lighting, 0.75f, nee, lighting);
	RT_SkyNEE(hit, rng, gi);
	VectorMA(lighting, 0.85f, gi, lighting);
	lighting[0] += 0.12f;
	lighting[1] += 0.12f;
	lighting[2] += 0.13f;
	if (rt_bounces->integer > 0 && bounce > 0) {
		RT_Bounce(hit, rng, 1, gi);
		lighting[0] += gi[0] * 0.55f;
		lighting[1] += gi[1] * 0.55f;
		lighting[2] += gi[2] * 0.55f;
	}
	color[0] = e->albedo[0] * lighting[0];
	color[1] = e->albedo[1] * lighting[1];
	color[2] = e->albedo[2] * lighting[2];
	(void)ray;
}

static void RT_Shade(const rt_ray_t *ray, const rt_hit_t *hit, unsigned *rng, int bounce, vec3_t color)
{
	vec3_t albedo, lighting, dl, nee, gi, emit;
	int flags;
	float emitScale;
	qboolean primaryHit;

	if (hit->sky) {
		RT_SkyColor(ray->dir, color);
		return;
	}
	if (hit->is_entity && hit->ent) {
		RT_ShadeEntity(ray, hit, rng, bounce, color);
		return;
	}
	if (!hit->surf || !hit->surf->texinfo) {
		VectorSet(color, 0.15f, 0.15f, 0.16f);
		return;
	}

	flags = hit->surf->texinfo->flags;
	RT_SampleTexture(hit->surf, hit->pos, albedo);

	if (flags & SURF_WARP && bounce >= 0) {
		rt_ray_t r2;
		rt_hit_t h2;
		vec3_t refl, water;
		float ndoti;

		ndoti = DotProduct(ray->dir, hit->normal);
		VectorMA(ray->dir, -2.0f * ndoti, hit->normal, r2.dir);
		VectorNormalize(r2.dir);
		VectorMA(hit->pos, RT_EPS, hit->normal, r2.origin);
		if (RT_Trace(r2.origin, r2.dir, 0.0f, RT_MAX_DIST, &h2))
			RT_Shade(&r2, &h2, rng, bounce - 1, refl);
		else
			RT_SkyColor(r2.dir, refl);
		water[0] = albedo[0] * 0.25f + refl[0] * 0.75f;
		water[1] = albedo[1] * 0.28f + refl[1] * 0.72f;
		water[2] = albedo[2] * 0.35f + refl[2] * 0.65f;
		VectorCopy(water, color);
		return;
	}

	VectorClear(lighting);
	primaryHit = (bounce >= rt_bounces->integer);
	{
		if (rt_mode->integer <= 1) {
			/* lightmaps + hybrid raytracing reflection rays */
			RT_SampleLightmap(hit->surf, hit->pos, lighting);
			RT_Dlights(hit, dl);
			VectorAdd(lighting, dl, lighting);
			if (primaryHit) {
				RT_NEE(hit, rng, nee);
				VectorMA(lighting, 0.85f, nee, lighting);
				RT_SkyNEE(hit, rng, gi);
				VectorAdd(lighting, gi, lighting);
			}
			if (rt_bounces->integer > 0 && bounce > 0) {
				RT_Bounce(hit, rng, 1, gi);
				lighting[0] += gi[0] * 0.72f;
				lighting[1] += gi[1] * 0.72f;
				lighting[2] += gi[2] * 0.72f;
			}
			lighting[0] += 0.08f;
			lighting[1] += 0.08f;
			lighting[2] += 0.09f;
		} else {
			if (primaryHit) {
				RT_NEE(hit, rng, nee);
				RT_SkyNEE(hit, rng, gi);
				VectorAdd(nee, gi, lighting);
			}
			RT_Dlights(hit, dl);
			VectorAdd(lighting, dl, lighting);
			lighting[0] += 0.06f;
			lighting[1] += 0.06f;
			lighting[2] += 0.07f;
			if (rt_bounces->integer > 0 && bounce > 0) {
				RT_Bounce(hit, rng, 1, gi);
				lighting[0] += gi[0];
				lighting[1] += gi[1];
				lighting[2] += gi[2];
			}
		}
	}

	if (flags & SURF_LIGHT) {
		emitScale = hit->surf->texinfo->value * 0.008f;
		if (emitScale < 1.0f)
			emitScale = 1.0f;
		VectorScale(albedo, emitScale, emit);
		VectorAdd(lighting, emit, lighting);
	}

	color[0] = albedo[0] * lighting[0];
	color[1] = albedo[1] * lighting[1];
	color[2] = albedo[2] * lighting[2];

	if (primaryHit && rt_reflect && rt_reflect->value > 0.01f
		&& !(flags & (SURF_SKY | SURF_NODRAW))) {
		vec3_t spec;
		float nv, fresnel, mix, gain, oneMinus;

		gain = rt_reflect->value;
		if (gain > 1.0f)
			gain = 1.0f;
		nv = -DotProduct(ray->dir, hit->normal);
		if (nv < 0.0f) nv = 0.0f;
		if (nv > 1.0f) nv = 1.0f;
		oneMinus = 1.0f - nv;
		fresnel = 0.08f + 0.92f * oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
		mix = gain * (0.28f + 0.72f * fresnel);
		if (mix > 0.82f)
			mix = 0.82f;
		RT_Glossy(ray, hit, rng, spec);
		color[0] = color[0] * (1.0f - mix * 0.55f) + spec[0] * (0.40f + 0.60f * albedo[0]) * mix;
		color[1] = color[1] * (1.0f - mix * 0.55f) + spec[1] * (0.40f + 0.60f * albedo[1]) * mix;
		color[2] = color[2] * (1.0f - mix * 0.55f) + spec[2] * (0.40f + 0.60f * albedo[2]) * mix;
	}
}

static void RT_Primary(int x, int y, unsigned *rng, vec3_t color, float *zeye)
{
	rt_ray_t ray;
	rt_hit_t hit;
	float nx, ny;

	nx = (2.0f * ((float)x + RT_Frand(rng)) / (float)rt_w - 1.0f) * rt_tanx;
	ny = (1.0f - 2.0f * ((float)y + RT_Frand(rng)) / (float)rt_h) * rt_tany;

	ray.origin[0] = r_origin[0];
	ray.origin[1] = r_origin[1];
	ray.origin[2] = r_origin[2];
	/* viewAxis[1] is left in this engine; the GL view matrix uses -axis[1]. */
	ray.dir[0] = viewAxis[0][0] - viewAxis[1][0] * nx + viewAxis[2][0] * ny;
	ray.dir[1] = viewAxis[0][1] - viewAxis[1][1] * nx + viewAxis[2][1] * ny;
	ray.dir[2] = viewAxis[0][2] - viewAxis[1][2] * nx + viewAxis[2][2] * ny;
	VectorNormalize(ray.dir);

	*zeye = skybox_farz;
	if (!RT_Trace(ray.origin, ray.dir, 0.0f, RT_MAX_DIST, &hit)) {
		__sync_fetch_and_add(&rt_stat_miss, 1);
		RT_SkyColor(ray.dir, color);
		return;
	}
	if (hit.sky)
		__sync_fetch_and_add(&rt_stat_sky, 1);
	else
		__sync_fetch_and_add(&rt_stat_hit, 1);
	{
		vec3_t delta;
		float z;
		VectorSubtract(hit.pos, r_origin, delta);
		z = DotProduct(delta, viewAxis[0]);
		if (z > 4.0f)
			*zeye = z;
	}
	RT_Shade(&ray, &hit, rng, rt_bounces->integer, color);
}

static void RT_PrimaryReflect(int x, int y, unsigned *rng, vec3_t color, float *alpha)
{
	rt_ray_t ray;
	rt_hit_t hit;
	float nx, ny, up;

	VectorClear(color);
	*alpha = 0.0f;

	nx = (2.0f * ((float)x + 0.5f) / (float)rt_w - 1.0f) * rt_tanx;
	ny = (1.0f - 2.0f * ((float)y + 0.5f) / (float)rt_h) * rt_tany;
	ray.origin[0] = r_origin[0];
	ray.origin[1] = r_origin[1];
	ray.origin[2] = r_origin[2];
	ray.dir[0] = viewAxis[0][0] - viewAxis[1][0] * nx + viewAxis[2][0] * ny;
	ray.dir[1] = viewAxis[0][1] - viewAxis[1][1] * nx + viewAxis[2][1] * ny;
	ray.dir[2] = viewAxis[0][2] - viewAxis[1][2] * nx + viewAxis[2][2] * ny;
	VectorNormalize(ray.dir);

	if (!RT_Trace(ray.origin, ray.dir, 0.0f, RT_MAX_DIST, &hit) || hit.sky || hit.is_entity)
		return;
	if (!hit.surf || !hit.surf->texinfo)
		return;

	{
		int flags = hit.surf->texinfo->flags;
		/* Water always reflects (entities included via RT_Trace). */
		if (flags & SURF_WARP) {
			RT_Shade(&ray, &hit, rng, rt_bounces->integer, color);
			*alpha = 0.65f;
			return;
		}
		/*
		 * Glossy overlay only on mostly-upward floors, and only when
		 * rt_reflect is meaningful — applying it to every wall caused
		 * fullscreen sparkle noise.
		 */
		up = hit.normal[2];
		if (hit.surf->flags & SURF_PLANEBACK)
			up = -up;
		if (up > 0.55f && !(flags & (SURF_SKY | SURF_NODRAW))) {
			vec3_t spec, albedo;
			float nv, fresnel, mix, gain, oneMinus;

			gain = (rt_reflect) ? rt_reflect->value : 0.0f;
			if (rt_mode && rt_mode->integer == 1 && gain < 0.35f)
				gain = 0.35f; /* hybrid default */
			if (gain < 0.05f)
				return;
			if (gain > 1.0f)
				gain = 1.0f;
			nv = -DotProduct(ray.dir, hit.normal);
			if (nv < 0.0f) nv = 0.0f;
			if (nv > 1.0f) nv = 1.0f;
			oneMinus = 1.0f - nv;
			fresnel = 0.04f + 0.96f * oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
			mix = gain * fresnel * 0.55f;
			if (mix < 0.02f)
				return;
			if (mix > 0.45f)
				mix = 0.45f;
			RT_SampleTexture(hit.surf, hit.pos, albedo);
			RT_Glossy(&ray, &hit, rng, spec);
			color[0] = spec[0] * (0.35f + 0.65f * albedo[0]);
			color[1] = spec[1] * (0.35f + 0.65f * albedo[1]);
			color[2] = spec[2] * (0.35f + 0.65f * albedo[2]);
			*alpha = mix;
		}
	}
}

static void *RT_Worker(void *arg)
{
	rt_job_t *job = (rt_job_t *)arg;
	int x, y;
	vec3_t color;
	unsigned seed = job->seed;

	for (y = job->y0; y < job->y1; y++) {
		for (x = 0; x < rt_w; x++) {
			int i = y * rt_w + x;
			float zeye = skybox_farz;
			if (job->mode == 1) {
				float alpha = 0.0f;
				RT_PrimaryReflect(x, y, &seed, color, &alpha);
				if (rt_frame_buf) {
					rt_frame_buf[i * 3 + 0] = color[0];
					rt_frame_buf[i * 3 + 1] = color[1];
					rt_frame_buf[i * 3 + 2] = color[2];
				}
				if (rt_zeye)
					rt_zeye[i] = alpha; /* reuse as alpha for reflection blit */
			} else {
				RT_Primary(x, y, &seed, color, &zeye);
				if (rt_frame_buf) {
					rt_frame_buf[i * 3 + 0] = color[0];
					rt_frame_buf[i * 3 + 1] = color[1];
					rt_frame_buf[i * 3 + 2] = color[2];
				}
				if (rt_zeye)
					rt_zeye[i] = zeye;
			}
		}
	}
	return NULL;
}

static void RT_EnsureBuffers(int w, int h)
{
	int size = w * h;
	if (w == rt_buf_w && h == rt_buf_h && rt_pixels && rt_accum_buf && rt_frame_buf && rt_zeye
		&& rt_refl_pixels)
		return;
	if (rt_pixels)
		Z_Free(rt_pixels);
	if (rt_accum_buf)
		Z_Free(rt_accum_buf);
	if (rt_frame_buf)
		Z_Free(rt_frame_buf);
	if (rt_zeye)
		Z_Free(rt_zeye);
	if (rt_refl_pixels)
		Z_Free(rt_refl_pixels);
	rt_pixels = Z_TagMalloc(size * 3, TAG_RENDER_IMAGE);
	rt_accum_buf = Z_TagMalloc(size * 3 * (int)sizeof(float), TAG_RENDER_IMAGE);
	rt_frame_buf = Z_TagMalloc(size * 3 * (int)sizeof(float), TAG_RENDER_IMAGE);
	rt_zeye = Z_TagMalloc(size * (int)sizeof(float), TAG_RENDER_IMAGE);
	rt_refl_pixels = Z_TagMalloc(size * 4, TAG_RENDER_IMAGE);
	memset(rt_pixels, 0, size * 3);
	memset(rt_accum_buf, 0, size * 3 * sizeof(float));
	memset(rt_frame_buf, 0, size * 3 * sizeof(float));
	memset(rt_zeye, 0, size * sizeof(float));
	memset(rt_refl_pixels, 0, size * 4);
	rt_buf_w = w;
	rt_buf_h = h;
	rt_buf_size = size;
	rt_first_frame = 1;
}

static void RT_TonemapBlit(float blend)
{
	int i, n = rt_w * rt_h;
	float keep = 1.0f - blend;
	float exposure = rt_exposure->value;
	int mode = rt_mode->integer;

	if (exposure < 0.1f)
		exposure = 0.1f;

	for (i = 0; i < n; i++) {
		float r = rt_frame_buf[i * 3 + 0];
		float g = rt_frame_buf[i * 3 + 1];
		float b = rt_frame_buf[i * 3 + 2];
		int ir, ig, ib;

		if (blend >= 0.999f) {
			rt_accum_buf[i * 3 + 0] = r;
			rt_accum_buf[i * 3 + 1] = g;
			rt_accum_buf[i * 3 + 2] = b;
		} else {
			r = rt_accum_buf[i * 3 + 0] = rt_accum_buf[i * 3 + 0] * keep + r * blend;
			g = rt_accum_buf[i * 3 + 1] = rt_accum_buf[i * 3 + 1] * keep + g * blend;
			b = rt_accum_buf[i * 3 + 2] = rt_accum_buf[i * 3 + 2] * keep + b * blend;
		}

		r *= exposure;
		g *= exposure;
		b *= exposure;
		if (mode >= 1) {
			/* path + hybrid overlay */
			r = r / (1.0f + r * 0.65f);
			g = g / (1.0f + g * 0.65f);
			b = b / (1.0f + b * 0.65f);
		} else {
			/* keep midtones bright; only compress blown highlights */
			if (r > 1.0f) r = 1.0f - 0.35f / r;
			if (g > 1.0f) g = 1.0f - 0.35f / g;
			if (b > 1.0f) b = 1.0f - 0.35f / b;
		}
		ir = (int)(r * 255.0f); clamp(ir, 0, 255);
		ig = (int)(g * 255.0f); clamp(ig, 0, 255);
		ib = (int)(b * 255.0f); clamp(ib, 0, 255);
		rt_pixels[i * 3 + 0] = (byte)ir;
		rt_pixels[i * 3 + 1] = (byte)ig;
		rt_pixels[i * 3 + 2] = (byte)ib;
	}
}

static void RT_DumpTGA(void)
{
	char path[MAX_OSPATH];
	FILE *f;
	byte header[18];
	int x, y;

	Com_sprintf(path, sizeof(path), "%s/rt_dump.tga", FS_Gamedir());
	f = fopen(path, "wb");
	if (!f) {
		Com_Printf("raytracer: could not write %s\n", path);
		return;
	}
	memset(header, 0, sizeof(header));
	header[2] = 2;
	header[12] = (byte)(rt_w & 255);
	header[13] = (byte)(rt_w >> 8);
	header[14] = (byte)(rt_h & 255);
	header[15] = (byte)(rt_h >> 8);
	header[16] = 24;
	fwrite(header, 1, 18, f);
	for (y = rt_h - 1; y >= 0; y--) {
		for (x = 0; x < rt_w; x++) {
			const byte *p = rt_pixels + (y * rt_w + x) * 3;
			byte bgr[3];
			bgr[0] = p[2];
			bgr[1] = p[1];
			bgr[2] = p[0];
			fwrite(bgr, 1, 3, f);
		}
	}
	fclose(f);
	Com_Printf("raytracer: wrote %s (%ix%i)\n", path, rt_w, rt_h);
}

static void RT_DrawQuadAlpha(float alpha)
{
	if (!rt_tex_ready) {
		qglGenTextures(1, &rt_texnum);
		rt_tex_ready = true;
	}

	GL_EnableMultitexture(false);
	GL_Bind(rt_texnum);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	qglTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rt_w, rt_h, 0, GL_RGB, GL_UNSIGNED_BYTE, rt_pixels);

	qglMatrixMode(GL_PROJECTION);
	qglPushMatrix();
	qglLoadIdentity();
	qglOrtho(0, r_newrefdef.width, r_newrefdef.height, 0, -99999, 99999);
	qglMatrixMode(GL_MODELVIEW);
	qglPushMatrix();
	qglLoadIdentity();

	qglDisable(GL_DEPTH_TEST);
	qglDisable(GL_CULL_FACE);
	qglDisable(GL_ALPHA_TEST);
	qglEnable(GL_TEXTURE_2D);
	if (alpha >= 0.999f) {
		qglDisable(GL_BLEND);
		GL_TexEnv(GL_REPLACE);
		qglColor4fv(colorWhite);
	} else {
		qglEnable(GL_BLEND);
		qglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		GL_TexEnv(GL_MODULATE);
		qglColor4f(1.0f, 1.0f, 1.0f, alpha);
	}

	qglBegin(GL_QUADS);
	qglTexCoord2f(0, 0);
	qglVertex2i(0, 0);
	qglTexCoord2f(1, 0);
	qglVertex2i(r_newrefdef.width, 0);
	qglTexCoord2f(1, 1);
	qglVertex2i(r_newrefdef.width, r_newrefdef.height);
	qglTexCoord2f(0, 1);
	qglVertex2i(0, r_newrefdef.height);
	qglEnd();

	qglColor4fv(colorWhite);
	qglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	qglDisable(GL_BLEND);
	qglEnable(GL_DEPTH_TEST);
	qglEnable(GL_CULL_FACE);

	qglMatrixMode(GL_PROJECTION);
	qglPopMatrix();
	qglMatrixMode(GL_MODELVIEW);
	qglPopMatrix();
}

static void RT_DrawQuad(void)
{
	RT_DrawQuadAlpha(1.0f);
}

static void RT_WriteDepth(void)
{
	int x, y;
	float zn = 4.0f;
	float zf = skybox_farz;
	float depth, a, b, sx, sy;

	if (!rt_zeye)
		return;
	if (zf < zn + 1.0f)
		zf = 4096.0f;
	depth = zf - zn;
	a = (zf + zn) / depth;
	b = (2.0f * zf * zn) / depth;
	sx = (float)r_newrefdef.width / (float)rt_w;
	sy = (float)r_newrefdef.height / (float)rt_h;

	qglMatrixMode(GL_PROJECTION);
	qglPushMatrix();
	qglLoadIdentity();
	qglOrtho(0, r_newrefdef.width, r_newrefdef.height, 0, -1, 1);
	qglMatrixMode(GL_MODELVIEW);
	qglPushMatrix();
	qglLoadIdentity();

	qglColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	qglDisable(GL_TEXTURE_2D);
	qglDisable(GL_BLEND);
	qglDisable(GL_CULL_FACE);
	qglDisable(GL_ALPHA_TEST);
	qglEnable(GL_DEPTH_TEST);
	qglDepthFunc(GL_ALWAYS);
	qglDepthMask(GL_TRUE);

	qglBegin(GL_QUADS);
	for (y = 0; y < rt_h; y++) {
		for (x = 0; x < rt_w; x++) {
			float z = rt_zeye[y * rt_w + x];
			float ndc, x0, y0, x1, y1;
			if (z < zn) z = zn;
			if (z > zf) z = zf;
			ndc = a - b / z;
			x0 = x * sx;
			y0 = y * sy;
			x1 = x0 + sx;
			y1 = y0 + sy;
			qglVertex3f(x0, y0, -ndc);
			qglVertex3f(x1, y0, -ndc);
			qglVertex3f(x1, y1, -ndc);
			qglVertex3f(x0, y1, -ndc);
		}
	}
	qglEnd();

	qglColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	qglDepthFunc(GL_LEQUAL);
	qglEnable(GL_CULL_FACE);
	qglEnable(GL_TEXTURE_2D);

	qglMatrixMode(GL_PROJECTION);
	qglPopMatrix();
	qglMatrixMode(GL_MODELVIEW);
	qglPopMatrix();
}

static void RT_DrawEntities(void)
{
	if (!r_drawentities->integer)
		return;
	if (!rt_weapons->integer) {
		entity_t *saved = currententity;
		entity_t *ent, *lastE;

		lastE = r_newrefdef.entities + r_newrefdef.num_entities;
		for (ent = r_newrefdef.entities; ent != lastE; ent++) {
			if (ent->flags & RF_WEAPONMODEL)
				ent->flags |= RF_VIEWERMODEL;
		}
		R_DrawEntitiesOnList();
		for (ent = r_newrefdef.entities; ent != lastE; ent++) {
			if ((ent->flags & (RF_WEAPONMODEL | RF_VIEWERMODEL)) == (RF_WEAPONMODEL | RF_VIEWERMODEL))
				ent->flags &= ~RF_VIEWERMODEL;
		}
		currententity = saved;
		return;
	}
	R_DrawEntitiesOnList();
}

qboolean R_RayTraceActive(void)
{
	return (r_raytrace && r_raytrace->integer
		&& r_worldmodel
		&& !(r_newrefdef.rdflags & RDF_NOWORLDMODEL));
}

qboolean R_RayLightmapMode(void)
{
	/* Modes 0 (lightmaps) and 1 (hybrid raytracing) keep the raster world. */
	return R_RayTraceActive() && (!rt_mode || rt_mode->integer <= 1);
}

qboolean R_RayPathMode(void)
{
	return R_RayTraceActive() && rt_mode && rt_mode->integer >= 2;
}

static void RT_FreeSurfaceCaches(void)
{
	int i;

	if (!rt_light_world)
		return;
	for (i = 0; i < rt_light_world->numsurfaces; i++) {
		msurface_t *s = rt_light_world->surfaces + i;
		if (s->rt_light) {
			Z_Free(s->rt_light);
			s->rt_light = NULL;
			s->rt_light_size = 0;
			s->rt_dirty = 0;
			s->rt_baked = 0;
		}
	}
}

static qboolean RT_EnsureSurfCache(msurface_t *surf, int size)
{
	int i;
	float scale;

	if (surf->rt_light && surf->rt_light_size == size)
		return true;
	if (surf->rt_light)
		Z_Free(surf->rt_light);
	surf->rt_light = Z_TagMalloc(size * 3 * (int)sizeof(float), TAG_RENDER_IMAGE);
	if (!surf->rt_light) {
		surf->rt_light_size = 0;
		return false;
	}
	surf->rt_light_size = size;
	surf->rt_dirty = 0;
	surf->rt_baked = 0;

	/* Seed from baked style-0 samples so partial surfaces match stock lighting */
	scale = gl_modulate ? gl_modulate->value : 1.0f;
	if (surf->samples) {
		for (i = 0; i < size; i++) {
			surf->rt_light[i * 3 + 0] = surf->samples[i * 3 + 0] * scale;
			surf->rt_light[i * 3 + 1] = surf->samples[i * 3 + 1] * scale;
			surf->rt_light[i * 3 + 2] = surf->samples[i * 3 + 2] * scale;
		}
	} else {
		memset(surf->rt_light, 0, size * 3 * sizeof(float));
	}
	return true;
}

static void RT_LuxelOrigin(const msurface_t *surf, int s, int t, vec3_t out)
{
	mtexinfo_t *tex = surf->texinfo;
	vec3_t facemid, normal;
	float area, starts, startt, mids, midt, ds, dt, len0, len1, dist;
	int i;

	RT_PolyStats(surf, facemid, &area);
	starts = (float)(surf->texturemins[0] + s * 16 + 8);
	startt = (float)(surf->texturemins[1] + t * 16 + 8);
	mids = DotProduct(facemid, tex->vecs[0]) + tex->vecs[0][3];
	midt = DotProduct(facemid, tex->vecs[1]) + tex->vecs[1][3];
	ds = starts - mids;
	dt = startt - midt;
	len0 = DotProduct(tex->vecs[0], tex->vecs[0]);
	len1 = DotProduct(tex->vecs[1], tex->vecs[1]);
	if (len0 < 1e-4f) len0 = 1.0f;
	if (len1 < 1e-4f) len1 = 1.0f;

	for (i = 0; i < 3; i++)
		out[i] = facemid[i] + tex->vecs[0][i] * (ds / len0) + tex->vecs[1][i] * (dt / len1);

	RT_SurfNormal(surf, normal);
	dist = DotProduct(out, surf->plane->normal) - surf->plane->dist;
	VectorMA(out, -dist, surf->plane->normal, out);
	VectorMA(out, RT_EPS * 2.0f, normal, out);
}

/*
 * True sky reach test (hit sky brush or escape world). Better than
 * RT_Occluded for roof holes — sky hits count as lit, not blocked.
 */
static qboolean RT_RayReachesSky(const vec3_t o, const vec3_t d, float tmax)
{
	rt_hit_t hit;

	if (!RT_Trace(o, d, RT_EPS, tmax, &hit))
		return true;
	return hit.sky;
}

/*
 * Returns sky visibility 0..1, or <0 if this luxel has no sky influence
 * (deep indoor — leave stock bake alone).
 */
static float RT_SkyVisibility(const vec3_t luxelPos, const vec3_t normal)
{
	vec3_t origin, dir, sun;
	float vis = 0.0f, weight = 0.0f, w, dist;
	int i, n, used;
	int order[16];

	VectorMA(luxelPos, RT_EPS * 3.0f, normal, origin);

	/* Straight up — roof openings */
	dir[0] = 0.0f; dir[1] = 0.0f; dir[2] = 1.0f;
	w = DotProduct(normal, dir);
	if (w > 0.02f) {
		weight += w * 2.0f;
		if (RT_RayReachesSky(origin, dir, RT_MAX_DIST))
			vis += w * 2.0f;
	}

	/* Sun + fixed soft samples */
	RT_SunDir(sun);
	w = DotProduct(normal, sun);
	if (w > 0.05f) {
		weight += w * 3.0f;
		if (RT_RayReachesSky(origin, sun, RT_MAX_DIST))
			vis += w * 3.0f;
	}
	{
		static const float jx[] = { 0.06f, -0.06f, 0.00f, 0.05f, -0.04f };
		static const float jy[] = { 0.00f, 0.00f, 0.06f, -0.05f, 0.04f };
		for (i = 0; i < 5; i++) {
			vec3_t s2;
			s2[0] = sun[0] + jx[i];
			s2[1] = sun[1] + jy[i];
			s2[2] = sun[2];
			VectorNormalize(s2);
			w = DotProduct(normal, s2);
			if (w <= 0.05f)
				continue;
			weight += w * 0.7f;
			if (RT_RayReachesSky(origin, s2, RT_MAX_DIST))
				vis += w * 0.7f;
		}
	}

	/* Nearest sky portals (by distance²) */
	n = rt_num_skylights;
	used = 0;
	for (i = 0; i < n; i++) {
		vec3_t d;
		float d2;
		int k, placed = 0;
		VectorSubtract(rt_skylights[i].center, origin, d);
		d2 = DotProduct(d, d);
		if (d2 > 4000.0f * 4000.0f)
			continue;
		for (k = 0; k < used; k++) {
			vec3_t od;
			float od2;
			VectorSubtract(rt_skylights[order[k]].center, origin, od);
			od2 = DotProduct(od, od);
			if (d2 < od2) {
				int m;
				if (used < 8) used++;
				for (m = used - 1; m > k; m--)
					order[m] = order[m - 1];
				order[k] = i;
				placed = 1;
				break;
			}
		}
		if (!placed && used < 8)
			order[used++] = i;
	}
	for (i = 0; i < used; i++) {
		const rt_light_t *L = &rt_skylights[order[i]];
		vec3_t ln;
		VectorSubtract(L->center, origin, dir);
		dist = VectorNormalize(dir);
		if (dist < 1.0f)
			continue;
		w = DotProduct(normal, dir);
		if (w <= 0.02f)
			continue;
		RT_SurfNormal(L->surf, ln);
		if (DotProduct(ln, dir) > 0.05f)
			continue;
		weight += w * 2.2f;
		if (RT_RayReachesSky(origin, dir, dist + 64.0f))
			vis += w * 2.2f;
	}

	if (weight < 1e-4f)
		return -1.0f;
	vis /= weight;
	if (vis < 0.0f) vis = 0.0f;
	if (vis > 1.0f) vis = 1.0f;
	return vis;
}

static float RT_EntityShadowFactor(const vec3_t luxelPos, const vec3_t normal)
{
	vec3_t origin, toE, hitPos, hitN;
	float factor = 1.0f;
	float dist, ndot;
	int i;

	VectorMA(luxelPos, RT_EPS * 2.0f, normal, origin);

	for (i = 0; i < rt_num_ents; i++) {
		const rt_ent_t *e = &rt_ents[i];
		float t;
		if (!e->valid)
			continue;
		VectorSubtract(e->origin, origin, toE);
		dist = VectorNormalize(toE);
		if (dist < 4.0f || dist > 128.0f)
			continue;
		ndot = DotProduct(normal, toE);
		if (ndot < 0.15f && normal[2] < 0.45f)
			continue;
		if (!RT_RayOrientedAABB(origin, toE, RT_EPS, dist + 4.0f, e, &t, hitPos, hitN))
			continue;
		{
			float s = 1.0f - (dist / 128.0f);
			if (s < 0.0f) s = 0.0f;
			factor *= 1.0f - 0.40f * s * s;
		}
	}
	if (factor < 0.40f)
		factor = 0.40f;
	return factor;
}

/*
 * Stable lightmap luxel shading — the "middle" look:
 * stock qrad bake + deterministic sky visibility (no Monte Carlo speckles).
 */
static void RT_ShadeLuxel(msurface_t *surf, int s, int t, vec3_t lighting)
{
	rt_hit_t hit;
	vec3_t baked, scol, sun, up;
	float mod, ent, vis, gain, keep, skyadd, m;
	int smax, idx, maps;
	byte *lm;

	memset(&hit, 0, sizeof(hit));
	hit.hit = true;
	hit.surf = surf;
	RT_LuxelOrigin(surf, s, t, hit.pos);
	RT_SurfNormal(surf, hit.normal);

	smax = (surf->extents[0] >> 4) + 1;
	idx = (t * smax + s) * 3;
	mod = gl_modulate ? gl_modulate->value : 1.0f;

	VectorClear(baked);
	lm = surf->samples;
	for (maps = 0; maps < MAXLIGHTMAPS && surf->styles[maps] != 255; maps++) {
		float sr = r_newrefdef.lightstyles[surf->styles[maps]].rgb[0];
		float sg = r_newrefdef.lightstyles[surf->styles[maps]].rgb[1];
		float sb = r_newrefdef.lightstyles[surf->styles[maps]].rgb[2];
		byte *blk = lm + maps * smax * ((surf->extents[1] >> 4) + 1) * 3;
		baked[0] += blk[idx + 0] * sr * mod;
		baked[1] += blk[idx + 1] * sg * mod;
		baked[2] += blk[idx + 2] * sb * mod;
	}

	vis = RT_SkyVisibility(hit.pos, hit.normal);
	ent = RT_EntityShadowFactor(hit.pos, hit.normal);

	if (vis < 0.0f) {
		/* Deep indoor — stock bake only */
		VectorScale(baked, ent, lighting);
		return;
	}

	/*
	 * Contrast like path-trace sky spill, but stable:
	 * open-to-sky luxels get warm/red sky add; occluded keep more bake.
	 */
	gain = (rt_skylight && rt_skylight->value > 0.0f) ? rt_skylight->value : 2.4f;
	keep = 0.40f + 0.45f * vis; /* more sky → slightly more bake retained too */
	skyadd = 55.0f * mod * (gain / 2.4f);
	if (rt_lightscale && rt_lightscale->value > 0.1f)
		skyadd *= rt_lightscale->value;

	up[0] = 0.0f; up[1] = 0.0f; up[2] = 1.0f;
	RT_SunDir(sun);
	RT_SkyColor(up, scol);
	lighting[0] = baked[0] * keep + vis * skyadd * (0.55f + 0.90f * scol[0]);
	lighting[1] = baked[1] * keep + vis * skyadd * (0.40f + 0.70f * scol[1]);
	lighting[2] = baked[2] * keep + vis * skyadd * (0.30f + 0.55f * scol[2]);
	/* Soft sun contribution */
	RT_SkyColor(sun, scol);
	lighting[0] += vis * skyadd * 0.25f * scol[0];
	lighting[1] += vis * skyadd * 0.22f * scol[1];
	lighting[2] += vis * skyadd * 0.16f * scol[2];

	VectorScale(lighting, ent, lighting);

	/* Clamp fireflies / purple blowouts into lightmap range */
	m = lighting[0];
	if (lighting[1] > m) m = lighting[1];
	if (lighting[2] > m) m = lighting[2];
	if (m > 320.0f) {
		float s = 320.0f / m;
		lighting[0] *= s;
		lighting[1] *= s;
		lighting[2] *= s;
	}
}

/*
 * Upload through the same TMU/path the world renderer uses (unit 1).
 * Leaving rt_dirty set also lets DrawWorld rebuild if this misses.
 */
static void RT_UploadDirtyLightmaps(void)
{
	int i;
	unsigned temp[128 * 128];
	int uploaded = 0;

	if (!r_worldmodel || !gl_state.lightmap_textures)
		return;

	GL_EnableMultitexture(true);

	for (i = 0; i < r_worldmodel->numsurfaces; i++) {
		msurface_t *surf = r_worldmodel->surfaces + i;
		int smax, tmax;

		if (!surf->rt_dirty || !surf->rt_light || !surf->samples || !surf->rt_baked)
			continue;
		if (!surf->texinfo || (surf->texinfo->flags & (SURF_SKY | SURF_WARP | SURF_TRANS33 | SURF_TRANS66)))
			continue;

		smax = (surf->extents[0] >> 4) + 1;
		tmax = (surf->extents[1] >> 4) + 1;
		if (smax < 1 || tmax < 1 || smax * tmax > (int)(sizeof(temp) / 4))
			continue;

		R_BuildLightMap(surf, (byte *)temp, smax * 4);
		R_SetCacheState(surf);

		GL_MBind(1, gl_state.lightmap_textures + surf->lightmaptexturenum);
		qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		qglTexSubImage2D(GL_TEXTURE_2D, 0,
			surf->light_s, surf->light_t,
			smax, tmax,
			GL_RGBA, GL_UNSIGNED_BYTE, temp);

		/* Leave dirty so DrawWorld also rebuilds once (belt and suspenders) */
		uploaded++;
	}

	GL_SelectTexture(0);
	GL_EnableMultitexture(false);

	if (uploaded && r_speeds->integer)
		Com_Printf("rt lm uploaded %i surfaces\n", uploaded);
}

static qboolean RT_SurfNearEntity(const msurface_t *surf)
{
	vec3_t center;
	float area, best = 99999.0f;
	int i;

	if (!surf->polys || rt_num_ents < 1)
		return false;
	RT_PolyStats(surf, center, &area);
	for (i = 0; i < rt_num_ents; i++) {
		vec3_t d;
		float dist;
		if (!rt_ents[i].valid)
			continue;
		VectorSubtract(rt_ents[i].origin, center, d);
		dist = VectorLength(d);
		if (dist < best)
			best = dist;
	}
	return best < 160.0f;
}

static void RT_BakeWholeSurface(msurface_t *surf)
{
	int smax, tmax, size, s, t;
	vec3_t sample;
	float *slot;

	smax = (surf->extents[0] >> 4) + 1;
	tmax = (surf->extents[1] >> 4) + 1;
	size = smax * tmax;
	if (size < 1 || !RT_EnsureSurfCache(surf, size))
		return;

	for (t = 0; t < tmax; t++) {
		for (s = 0; s < smax; s++) {
			RT_ShadeLuxel(surf, s, t, sample);
			slot = surf->rt_light + (t * smax + s) * 3;
			slot[0] = sample[0];
			slot[1] = sample[1];
			slot[2] = sample[2];
		}
	}
	surf->rt_baked = 1;
	surf->rt_dirty = 1;
}

static void RT_InvalidateLightmapBakes(void)
{
	int i;

	if (!r_worldmodel)
		return;
	for (i = 0; i < r_worldmodel->numsurfaces; i++) {
		msurface_t *s = r_worldmodel->surfaces + i;
		s->rt_baked = 0;
		s->rt_dirty = 0;
	}
	rt_lm_surf_cursor = 0;
	rt_lm_cursor = 0;
}

static void RT_UpdateLightmaps(void)
{
	int i, budget, scale, processed, baked_surfs, guard;
	const int ALGO = 7; /* deterministic sky+bake middle path */

	if (!r_worldmodel)
		return;

	if (rt_lm_algo != ALGO) {
		rt_lm_algo = ALGO;
		RT_InvalidateLightmapBakes();
		RT_RunLog("raytracer: rebaking lightmaps (algo %i, stable sky GI)\n", ALGO);
	}

	if (rt_light_world != r_worldmodel)
		RT_CollectLights();

	scale = rt_scale ? rt_scale->integer : 4;
	if (scale < 1) scale = 1;
	if (scale > 8) scale = 8;
	/* Aggressive nearby budget so the courtyard settles in seconds */
	budget = 64 + (8 - scale) * 20;
	if (budget < 48) budget = 48;
	if (budget > 200) budget = 200;

	processed = 0;
	baked_surfs = 0;
	guard = r_worldmodel->numsurfaces * 4 + 16;

	while (baked_surfs < budget && guard-- > 0) {
		msurface_t *surf;
		vec3_t center, delta;
		float area, dist;
		int smax, tmax, size;

		if (rt_lm_surf_cursor < 0 || rt_lm_surf_cursor >= r_worldmodel->numsurfaces)
			rt_lm_surf_cursor = 0;
		surf = r_worldmodel->surfaces + rt_lm_surf_cursor;
		rt_lm_surf_cursor++;

		if (!surf->samples || !surf->texinfo || !surf->polys)
			continue;
		if (surf->texinfo->flags & (SURF_SKY | SURF_WARP | SURF_TRANS33 | SURF_TRANS66 | SURF_NODRAW))
			continue;
		if (surf->rt_baked)
			continue; /* bake once — stable */

		RT_PolyStats(surf, center, &area);
		VectorSubtract(center, r_origin, delta);
		dist = VectorLength(delta);
		/* Prefer near/visible; still allow farther so the map fills in */
		if (dist > 2200.0f && surf->visframe < r_framecount - 2)
			continue;

		smax = (surf->extents[0] >> 4) + 1;
		tmax = (surf->extents[1] >> 4) + 1;
		size = smax * tmax;
		processed += size;
		RT_BakeWholeSurface(surf);
		surf->cached_light[0] = -999.0f;
		baked_surfs++;
	}

	RT_UploadDirtyLightmaps();

	{
		static int report_ms;
		int now = Sys_Milliseconds();
		if (r_speeds->integer || ((rt_log && rt_log->integer) && now - report_ms > 2000)) {
			int baked = 0;
			for (i = 0; i < r_worldmodel->numsurfaces; i++)
				if (r_worldmodel->surfaces[i].rt_baked)
					baked++;
			RT_RunLog("rt lm: CPU  baked %i/%i  frame %i  skyportals %i  scale %i\n",
				baked, r_worldmodel->numsurfaces, baked_surfs, rt_num_skylights,
				rt_scale ? rt_scale->integer : 0);
			report_ms = now;
		}
	}
}

void R_RayApplySurfaceLight(const msurface_t *surf, float *blocklights, int size)
{
	int i;
	const float *src;

	if (!surf || !blocklights || !surf->rt_light || surf->rt_light_size != size || !surf->rt_baked)
		return;

	src = surf->rt_light;
	for (i = 0; i < size; i++, blocklights += 3, src += 3) {
		blocklights[0] = src[0];
		blocklights[1] = src[1];
		blocklights[2] = src[2];
	}
}

qboolean R_RaySoftDlights(const msurface_t *surf, float *blocklights)
{
	int lnum, s, t, smax, tmax;
	float *bl;
	vec3_t normal, luxelPos, origin, toL;
	float frad, fdist, fminlight, dist, ndotl, att;

	if (!surf || !blocklights || !R_RayLightmapMode())
		return false;
	if (!r_newrefdef.num_dlights)
		return true;

	smax = (surf->extents[0] >> 4) + 1;
	tmax = (surf->extents[1] >> 4) + 1;
	RT_SurfNormal(surf, normal);
	bl = blocklights;

	for (t = 0; t < tmax; t++) {
		for (s = 0; s < smax; s++, bl += 3) {
			RT_LuxelOrigin(surf, s, t, luxelPos);
			VectorMA(luxelPos, RT_EPS, normal, origin);

			for (lnum = 0; lnum < r_newrefdef.num_dlights; lnum++) {
				dlight_t *dl;

				if (!(surf->dlightbits & (1 << lnum)))
					continue;
				dl = &r_newrefdef.dlights[lnum];
				frad = dl->intensity;
				VectorSubtract(dl->origin, luxelPos, toL);
				dist = VectorLength(toL);
				if (dist < 1.0f)
					dist = 1.0f;
				fdist = dist;
				frad -= fdist;
				fminlight = 0.0f;
				if (frad <= fminlight)
					continue;
				ndotl = DotProduct(toL, normal) / dist;
				if (ndotl <= 0.05f)
					continue;
				toL[0] /= dist;
				toL[1] /= dist;
				toL[2] /= dist;
				if (RT_Occluded(origin, toL, dist))
					continue;
				att = frad * ndotl;
				bl[0] += att * dl->color[0];
				bl[1] += att * dl->color[1];
				bl[2] += att * dl->color[2];
			}
		}
	}
	return true;
}

static int RT_ThreadCount(int limit)
{
	int threads = rt_threads->integer;
	if (threads <= 0)
		threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (threads < 1) threads = 1;
	if (threads > RT_MAX_THREADS) threads = RT_MAX_THREADS;
	if (limit > 0 && threads > limit) threads = limit;
	return threads;
}

static void RT_RunWorkers(int mode)
{
	pthread_t th[RT_MAX_THREADS];
	rt_job_t jobs[RT_MAX_THREADS];
	int started[RT_MAX_THREADS];
	int i, threads;

	threads = RT_ThreadCount(rt_h);
	for (i = 0; i < threads; i++) {
		jobs[i].y0 = (rt_h * i) / threads;
		jobs[i].y1 = (rt_h * (i + 1)) / threads;
		jobs[i].seed = rt_frame_seed + (unsigned)i * 7919u + 1u;
		jobs[i].mode = mode;
		started[i] = (pthread_create(&th[i], NULL, RT_Worker, &jobs[i]) == 0);
		if (!started[i])
			RT_Worker(&jobs[i]);
	}
	for (i = 0; i < threads; i++) {
		if (started[i])
			pthread_join(th[i], NULL);
	}
}

static void RT_SetupViewRays(void)
{
	int scale, max_w, max_h;
	qboolean path_mode, hybrid;

	scale = rt_scale->integer;
	if (scale < 1) scale = 1;
	if (scale > 8) scale = 8;

	path_mode = R_RayPathMode();
	hybrid = (rt_mode && rt_mode->integer == 1);
	/*
	 * Lightmap bake uses high scale as a cheaper budget. Path-trace used the
	 * same cvar and collapsed to 160x128 at "low" — remap so path stays readable.
	 * Hybrid uses a medium overlay res so GI is visible without full replace.
	 */
	if (path_mode) {
		/* Cap for playable framerate on Metal */
		if (scale < 3)
			scale = 3;
		if (scale >= 8)
			scale = 4;
		else if (scale >= 6)
			scale = 3;
		max_w = 480;
		max_h = 270;
	} else if (hybrid) {
		/* Cheap reflection pass resolution */
		if (scale < 4)
			scale = 4;
		max_w = 320;
		max_h = 180;
	} else {
		max_w = 400;
		max_h = 240;
	}

	rt_w = r_newrefdef.width / scale;
	rt_h = r_newrefdef.height / scale;
	if (rt_w < 40) rt_w = 40;
	if (rt_h < 22) rt_h = 22;
	if (rt_w > max_w) rt_w = max_w;
	if (rt_h > max_h) rt_h = max_h;
	rt_tanx = tanf(r_newrefdef.fov_x * M_PI_DIV_360);
	rt_tany = tanf(r_newrefdef.fov_y * M_PI_DIV_360);
	rt_frame_seed += 113;
	RT_EnsureBuffers(rt_w, rt_h);
}

static void RT_DrawReflectQuad(void)
{
	int i, n = rt_w * rt_h;
	float exposure = rt_exposure->value;
	float blend = 0.35f;
	static vec3_t prev_org;
	static vec3_t prev_fwd;
	static int refl_init = 1;
	vec3_t delta;
	float moved, rot;

	if (exposure < 0.1f)
		exposure = 0.1f;
	if (!rt_refl_pixels || !rt_accum_buf)
		return;

	VectorSubtract(r_origin, prev_org, delta);
	moved = VectorLength(delta);
	rot = 1.0f - DotProduct(viewAxis[0], prev_fwd);
	if (refl_init || moved > 6.0f || rot > 0.015f)
		blend = 1.0f;
	else if (moved > 1.0f || rot > 0.002f)
		blend = 0.55f;
	refl_init = 0;
	VectorCopy(r_origin, prev_org);
	VectorCopy(viewAxis[0], prev_fwd);

	for (i = 0; i < n; i++) {
		float r = rt_frame_buf[i * 3 + 0];
		float g = rt_frame_buf[i * 3 + 1];
		float b = rt_frame_buf[i * 3 + 2];
		float a = rt_zeye ? rt_zeye[i] : 0.0f;
		float ar, ag, ab;
		int ir, ig, ib, ia;

		ar = rt_accum_buf[i * 3 + 0] = rt_accum_buf[i * 3 + 0] * (1.0f - blend) + r * blend;
		ag = rt_accum_buf[i * 3 + 1] = rt_accum_buf[i * 3 + 1] * (1.0f - blend) + g * blend;
		ab = rt_accum_buf[i * 3 + 2] = rt_accum_buf[i * 3 + 2] * (1.0f - blend) + b * blend;

		r = ar * exposure;
		g = ag * exposure;
		b = ab * exposure;
		if (r > 1.0f) r = 1.0f - 0.35f / r;
		if (g > 1.0f) g = 1.0f - 0.35f / g;
		if (b > 1.0f) b = 1.0f - 0.35f / b;
		if (a < 0.0f) a = 0.0f;
		if (a > 1.0f) a = 1.0f;
		/* Soften alpha so sparse glossy hits don't sparkle */
		a *= blend < 1.0f ? 0.75f : 0.55f;
		ir = (int)(r * 255.0f); clamp(ir, 0, 255);
		ig = (int)(g * 255.0f); clamp(ig, 0, 255);
		ib = (int)(b * 255.0f); clamp(ib, 0, 255);
		ia = (int)(a * 255.0f); clamp(ia, 0, 255);
		rt_refl_pixels[i * 4 + 0] = (byte)ir;
		rt_refl_pixels[i * 4 + 1] = (byte)ig;
		rt_refl_pixels[i * 4 + 2] = (byte)ib;
		rt_refl_pixels[i * 4 + 3] = (byte)ia;
	}

	if (!rt_refl_tex_ready) {
		qglGenTextures(1, &rt_refl_texnum);
		rt_refl_tex_ready = true;
	}

	GL_EnableMultitexture(false);
	GL_Bind(rt_refl_texnum);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	qglTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	qglTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rt_w, rt_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rt_refl_pixels);

	qglMatrixMode(GL_PROJECTION);
	qglPushMatrix();
	qglLoadIdentity();
	qglOrtho(0, r_newrefdef.width, r_newrefdef.height, 0, -99999, 99999);
	qglMatrixMode(GL_MODELVIEW);
	qglPushMatrix();
	qglLoadIdentity();

	qglDisable(GL_DEPTH_TEST);
	qglDisable(GL_CULL_FACE);
	qglDisable(GL_ALPHA_TEST);
	qglEnable(GL_BLEND);
	/* Alpha blend, not additive — additive made floor sparkle */
	qglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	qglEnable(GL_TEXTURE_2D);
	GL_TexEnv(GL_MODULATE);
	qglColor4f(1, 1, 1, 1);

	qglBegin(GL_QUADS);
	qglTexCoord2f(0, 0);
	qglVertex2i(0, 0);
	qglTexCoord2f(1, 0);
	qglVertex2i(r_newrefdef.width, 0);
	qglTexCoord2f(1, 1);
	qglVertex2i(r_newrefdef.width, r_newrefdef.height);
	qglTexCoord2f(0, 1);
	qglVertex2i(0, r_newrefdef.height);
	qglEnd();

	qglBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	qglDisable(GL_BLEND);
	qglEnable(GL_DEPTH_TEST);
	qglEnable(GL_CULL_FACE);

	qglMatrixMode(GL_PROJECTION);
	qglPopMatrix();
	qglMatrixMode(GL_MODELVIEW);
	qglPopMatrix();
}

void R_RayPostFrame(void)
{
	float gain;
	static int report_ms;

	if (!R_RayLightmapMode() || R_RayPathMode())
		return;

	/*
	 * Hybrid raytracing = same stable lightmaps + cheap glossy/water
	 * reflections (no fullscreen path overlay — that was too slow).
	 * Lightmaps mode only reflects when rt_reflect is cranked up.
	 */
	if (rt_mode && rt_mode->integer == 1)
		gain = (rt_reflect && rt_reflect->value > 0.05f) ? rt_reflect->value : 0.35f;
	else
		gain = (rt_reflect) ? rt_reflect->value : 0.0f;
	if (gain < 0.15f)
		return;

	RT_SetupViewRays();
	RT_RunWorkers(1);
	RT_DrawReflectQuad();

	{
		int now = Sys_Milliseconds();
		if ((rt_log && rt_log->integer) && now - report_ms > 2000) {
			RT_RunLog("rt hybrid: reflect overlay  %ix%i  gain %.2f\n",
				rt_w, rt_h, gain);
			report_ms = now;
		}
	}
}

static qboolean RT_PathTraceFrame(void)
{
	int t0, t1;
	float blend, moved, rot;
	qboolean used_gpu = false;

	RT_SetupViewRays();

	rt_stat_hit = rt_stat_sky = rt_stat_miss = 0;
	t0 = Sys_Milliseconds();
#ifdef __APPLE__
	if (rt_gpu && rt_gpu->integer && RT_GPU_Active() && rt_frame_buf) {
		int spp = rt_spp ? rt_spp->integer : 1;
		int pixels = rt_w * rt_h;
		/* Prefer speed; temporal accum cleans noise while standing still */
		if (pixels > 80000 && spp > 1)
			spp = 1;
		else if (spp > 2)
			spp = 2;
		if (spp < 1) spp = 1;
		used_gpu = RT_GPU_PathTrace(rt_w, rt_h, rt_tanx, rt_tany,
			r_origin, viewAxis[0], viewAxis[1], viewAxis[2],
			rt_skylight ? rt_skylight->value : 2.4f,
			rt_lightscale ? rt_lightscale->value : 1.35f,
			rt_bounces ? rt_bounces->integer : 0,
			spp,
			rt_frame_seed,
			rt_frame_buf, rt_zeye);
	}
#endif
	if (!used_gpu)
		RT_RunWorkers(0);
	t1 = Sys_Milliseconds();

	blend = 1.0f;
	if (rt_accum->integer && !rt_first_frame) {
		vec3_t delta;
		VectorSubtract(r_origin, rt_prev_org, delta);
		moved = VectorLength(delta);
		rot = 1.0f - DotProduct(viewAxis[0], rt_prev_fwd);
		if (moved > 8.0f || rot > 0.02f)
			blend = 1.0f;
		else if (moved > 1.0f || rot > 0.002f)
			blend = 0.28f;
		else
			blend = 0.04f; /* settle faster while standing still */
	}
	rt_first_frame = 0;
	VectorCopy(r_origin, rt_prev_org);
	VectorCopy(viewAxis[0], rt_prev_fwd);

	RT_TonemapBlit(blend);
	RT_DrawQuad();
	RT_WriteDepth();
	RT_DrawEntities();

	if (rt_dump->integer) {
		static int dump_frames;
		dump_frames++;
		if (dump_frames == 6) {
			RT_DumpTGA();
			if (rt_dump->integer >= 2)
				Cbuf_AddText("quit\n");
		}
	}

	{
		static int report_ms;
		int now = Sys_Milliseconds();
		if (r_speeds->integer || ((rt_log && rt_log->integer) && now - report_ms > 2000)) {
			RT_RunLog("rt path: %s  %ix%i  %i ms  spp %i  lights %i  ents %i  bounce %i  scale %i\n",
				used_gpu ? "GPU" : "CPU",
				rt_w, rt_h, t1 - t0,
				rt_spp ? rt_spp->integer : 1,
				rt_num_lights, rt_num_ents,
				rt_bounces ? rt_bounces->integer : 0,
				rt_scale ? rt_scale->integer : 0);
			report_ms = now;
		}
	}

	return true;
}

static void RT_ForceStockLightmaps(void)
{
	int i;

	if (!r_worldmodel)
		return;
	for (i = 0; i < r_worldmodel->numsurfaces; i++) {
		msurface_t *s = r_worldmodel->surfaces + i;
		if (!s->samples)
			continue;
		s->rt_baked = 0;
		s->rt_dirty = 1; /* force one rebuild from BSP samples */
		s->cached_light[0] = -999.0f;
	}
	rt_lm_algo = 0;
}

qboolean R_RayTraceFrame(void)
{
	static qboolean warned;
	static qboolean was_active;

	if (!R_RayTraceActive()) {
		if (was_active) {
			RT_ForceStockLightmaps();
			Com_Printf("raytracer: restored stock lightmaps\n");
			was_active = false;
		}
		return false;
	}
	was_active = true;

	if (!warned) {
		{
			const char *modename = "lightmaps";
			if (rt_mode && rt_mode->integer == 1)
				modename = "raytracing";
			else if (rt_mode && rt_mode->integer >= 2)
				modename = "path-trace";
			RT_RunLog("raytracer: active  mode=%s  gpu=%s  scale=%i  bounces=%i  skylight=%.2f  lightscale=%.2f\n",
				modename,
#ifdef __APPLE__
				(rt_gpu && rt_gpu->integer && RT_GPU_Active()) ? "Metal" : "off",
#else
				"off",
#endif
				rt_scale ? rt_scale->integer : 0,
				rt_bounces ? rt_bounces->integer : 0,
				rt_skylight ? rt_skylight->value : 0.0f,
				rt_lightscale ? rt_lightscale->value : 0.0f);
		}
		warned = true;
	}

	if (rt_light_world != r_worldmodel) {
		RT_FreeSurfaceCaches();
		RT_CollectLights();
		rt_lm_cursor = 0;
		rt_lm_surf_cursor = 0;
		rt_lm_algo = 0;
#ifdef __APPLE__
		RT_GPU_InvalidateWorld();
#endif
	}

	RT_CollectEntities();

	/* R_DrawWorld normally owns currententity; R_LightPoint needs it. */
	{
		static entity_t rt_worldent;
		memset(&rt_worldent, 0, sizeof(rt_worldent));
		VectorCopy(r_newrefdef.vieworg, rt_worldent.origin);
		currententity = &rt_worldent;
	}

	if (R_RayPathMode())
		return RT_PathTraceFrame();

	/* lightmaps + hybrid raytracing: bake luxels, keep raster world */
	RT_UpdateLightmaps();
	return false;
}

void R_RayInit(void)
{
	r_raytrace = Cvar_Get("r_raytrace", "1", CVAR_ARCHIVE);
	rt_scale = Cvar_Get("rt_scale", "4", CVAR_ARCHIVE);
	rt_mode = Cvar_Get("rt_mode", "0", CVAR_ARCHIVE); /* lightmaps = middle path */
	rt_bounces = Cvar_Get("rt_bounces", "0", CVAR_ARCHIVE);
	rt_accum = Cvar_Get("rt_accum", "1", CVAR_ARCHIVE);
	rt_exposure = Cvar_Get("rt_exposure", "2.6", CVAR_ARCHIVE);
	rt_lightscale = Cvar_Get("rt_lightscale", "1.35", CVAR_ARCHIVE);
	rt_skylight = Cvar_Get("rt_skylight", "2.4", CVAR_ARCHIVE);
	rt_reflect = Cvar_Get("rt_reflect", "0", CVAR_ARCHIVE);
	rt_threads = Cvar_Get("rt_threads", "0", CVAR_ARCHIVE);
	rt_weapons = Cvar_Get("rt_weapons", "1", CVAR_ARCHIVE);
	rt_dump = Cvar_Get("rt_dump", "0", 0);
#ifdef __APPLE__
	rt_gpu = Cvar_Get("rt_gpu", "1", CVAR_ARCHIVE);
#else
	rt_gpu = Cvar_Get("rt_gpu", "0", CVAR_ARCHIVE);
#endif
	rt_spp = Cvar_Get("rt_spp", "1", CVAR_ARCHIVE);
	rt_log = Cvar_Get("rt_log", "1", CVAR_ARCHIVE);
	rt_tex_ready = false;
	rt_texnum = 0;
	rt_refl_tex_ready = false;
	rt_refl_texnum = 0;
	rt_first_frame = 1;
	rt_lm_cursor = 0;
	rt_lm_surf_cursor = 0;
	rt_runlog_frame = 0;
#ifdef __APPLE__
	if (rt_gpu->integer) {
		if (RT_GPU_Init())
			RT_RunLog("raytracer: Metal GPU init OK\n");
		else
			RT_RunLog("raytracer: Metal GPU init FAILED — using CPU\n");
	} else {
		RT_RunLog("raytracer: rt_gpu 0 — CPU tracer\n");
	}
#else
	RT_RunLog("raytracer: CPU tracer (non-Apple build)\n");
#endif
}

void R_RayShutdown(void)
{
	if (rt_runlog) {
		fprintf(rt_runlog, "======== RT session end ========\n");
		fclose(rt_runlog);
		rt_runlog = NULL;
	}
#ifdef __APPLE__
	RT_GPU_Shutdown();
#endif
	RT_FreeSurfaceCaches();
	if (rt_pixels) {
		Z_Free(rt_pixels);
		rt_pixels = NULL;
	}
	if (rt_accum_buf) {
		Z_Free(rt_accum_buf);
		rt_accum_buf = NULL;
	}
	if (rt_frame_buf) {
		Z_Free(rt_frame_buf);
		rt_frame_buf = NULL;
	}
	if (rt_zeye) {
		Z_Free(rt_zeye);
		rt_zeye = NULL;
	}
	if (rt_refl_pixels) {
		Z_Free(rt_refl_pixels);
		rt_refl_pixels = NULL;
	}
	rt_buf_size = 0;
	if (rt_tex_ready) {
		qglDeleteTextures(1, &rt_texnum);
		rt_tex_ready = false;
	}
	if (rt_refl_tex_ready) {
		qglDeleteTextures(1, &rt_refl_texnum);
		rt_refl_tex_ready = false;
	}
	rt_num_lights = 0;
	rt_num_skylights = 0;
	rt_num_ents = 0;
	rt_light_world = NULL;
}
