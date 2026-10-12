/*
 * CPU BSP path tracer for this OpenGL 1.x Quake II client.
 *
 * NVIDIA Q2RTX / Christoph Schied's Q2VKPT replace the rasterizer with a
 * hardware path tracer: primary rays, next-event estimation toward
 * emissive surfaces, one indirect bounce, then temporal accumulation.
 * Those projects need Vulkan ray tracing, which this Mac / renderer do not
 * have. Early Q2VKPT work (and older Quake RT experiments) did the same
 * algorithm on the CPU, using the map BSP as the acceleration structure
 * instead of a GPU BLAS/TLAS.
 *
 * This file follows that software path: walk the render BSP per ray,
 * sample textures + lightmaps, shadow-test dlights, bounce once for GI
 * and water reflections, accumulate over frames, blit as a textured quad.
 *
 * Console:
 *   r_raytrace 1     enable (default on)
 *   rt_scale 3       internal resolution divisor
 *   rt_mode 0        0 = hybrid (lightmaps + real shadows/GI)
 *                    1 = path trace (ignore baked lightmaps)
 *   rt_bounces 1     0 or 1 extra bounce
 *   rt_accum 1       temporal filter
 */
#include "gl_local.h"

#include <pthread.h>
#include <unistd.h>

#define RT_MAX_DIST		16384.0f
#define RT_EPS			0.125f
#define RT_MAX_LIGHTS	384
#define RT_MAX_THREADS	16
#define RT_STACK		72

typedef struct {
	qboolean	hit;
	qboolean	sky;
	float		t;
	vec3_t		pos;
	vec3_t		normal;
	msurface_t	*surf;
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

static byte		*rt_pixels;
static float	*rt_frame_buf;
static float	*rt_accum_buf;
static float	*rt_zeye;
static int		rt_buf_w, rt_buf_h, rt_buf_size;

extern float	skybox_farz;
static GLuint	rt_texnum;
static qboolean	rt_tex_ready;

static rt_light_t	rt_lights[RT_MAX_LIGHTS];
static int			rt_num_lights;
static float		rt_light_sum;
static rt_light_t	rt_skylights[RT_MAX_LIGHTS];
static int			rt_num_skylights;
static const bspModel_t	*rt_light_world;

static vec3_t	rt_prev_org;
static vec3_t	rt_prev_fwd;
static int		rt_first_frame = 1;

static float	rt_tanx, rt_tany;
static int		rt_w, rt_h;
static unsigned	rt_frame_seed;
static int		rt_stat_hit, rt_stat_sky, rt_stat_miss;

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
	float h, sun;
	vec3_t sunDir;

	RT_SunDir(sunDir);
	h = dir[2] * 0.5f + 0.5f;
	out[0] = 0.55f + 0.55f * h;
	out[1] = 0.58f + 0.48f * h;
	out[2] = 0.62f + 0.42f * h;
	sun = DotProduct(dir, sunDir);
	if (sun > 0.0f) {
		sun = sun * sun;
		sun = sun * sun;
		sun = sun * sun;
		out[0] += sun * 6.5f;
		out[1] += sun * 5.4f;
		out[2] += sun * 3.6f;
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

static qboolean RT_Trace(const vec3_t o, const vec3_t d, float tmin, float tmax, rt_hit_t *hit)
{
	memset(hit, 0, sizeof(*hit));
	hit->t = tmax;
	if (!r_worldmodel)
		return false;
	RT_Traverse(r_worldmodel->nodes, o, d, tmin, tmax, hit);
	return hit->hit;
}

static qboolean RT_Occluded(const vec3_t o, const vec3_t d, float tmax)
{
	rt_hit_t hit;
	return RT_Trace(o, d, RT_EPS, tmax - RT_EPS * 2.0f, &hit) && !hit.sky;
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
		if (rt_mode->integer == 0) {
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
	return NULL;
}

static void RT_EnsureBuffers(int w, int h)
{
	int size = w * h;
	if (w == rt_buf_w && h == rt_buf_h && rt_pixels && rt_accum_buf && rt_frame_buf && rt_zeye)
		return;
	if (rt_pixels)
		Z_Free(rt_pixels);
	if (rt_accum_buf)
		Z_Free(rt_accum_buf);
	if (rt_frame_buf)
		Z_Free(rt_frame_buf);
	if (rt_zeye)
		Z_Free(rt_zeye);
	rt_pixels = Z_TagMalloc(size * 3, TAG_RENDER_IMAGE);
	rt_accum_buf = Z_TagMalloc(size * 3 * (int)sizeof(float), TAG_RENDER_IMAGE);
	rt_frame_buf = Z_TagMalloc(size * 3 * (int)sizeof(float), TAG_RENDER_IMAGE);
	rt_zeye = Z_TagMalloc(size * (int)sizeof(float), TAG_RENDER_IMAGE);
	memset(rt_pixels, 0, size * 3);
	memset(rt_accum_buf, 0, size * 3 * sizeof(float));
	memset(rt_frame_buf, 0, size * 3 * sizeof(float));
	memset(rt_zeye, 0, size * sizeof(float));
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
		if (mode != 0) {
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

static void RT_DrawQuad(void)
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
	qglDisable(GL_BLEND);
	qglEnable(GL_TEXTURE_2D);
	GL_TexEnv(GL_REPLACE);
	qglColor4fv(colorWhite);

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

	qglEnable(GL_DEPTH_TEST);
	qglEnable(GL_CULL_FACE);

	qglMatrixMode(GL_PROJECTION);
	qglPopMatrix();
	qglMatrixMode(GL_MODELVIEW);
	qglPopMatrix();
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

qboolean R_RayTraceFrame(void)
{
	int scale, threads, i, t0, t1;
	float blend, moved, rot;
	pthread_t th[RT_MAX_THREADS];
	rt_job_t jobs[RT_MAX_THREADS];
	int started[RT_MAX_THREADS];
	static qboolean warned;

	if (!R_RayTraceActive())
		return false;

	if (!warned) {
		Com_Printf("raytracer: CPU BSP path tracer (Q2VKPT-style). r_raytrace 0 disables.\n");
		warned = true;
	}

	if (rt_light_world != r_worldmodel)
		RT_CollectLights();

	/* R_DrawWorld normally owns currententity; R_LightPoint needs it. */
	{
		static entity_t rt_worldent;
		memset(&rt_worldent, 0, sizeof(rt_worldent));
		VectorCopy(r_newrefdef.vieworg, rt_worldent.origin);
		currententity = &rt_worldent;
	}

	scale = rt_scale->integer;
	if (scale < 1) scale = 1;
	if (scale > 8) scale = 8;
	rt_w = r_newrefdef.width / scale;
	rt_h = r_newrefdef.height / scale;
	if (rt_w < 40) rt_w = 40;
	if (rt_h < 22) rt_h = 22;
	if (rt_w > 400) rt_w = 400;
	if (rt_h > 240) rt_h = 240;

	rt_tanx = tanf(r_newrefdef.fov_x * M_PI_DIV_360);
	rt_tany = tanf(r_newrefdef.fov_y * M_PI_DIV_360);
	rt_frame_seed += 113;
	RT_EnsureBuffers(rt_w, rt_h);

	threads = rt_threads->integer;
	if (threads <= 0)
		threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
	if (threads < 1) threads = 1;
	if (threads > RT_MAX_THREADS) threads = RT_MAX_THREADS;
	if (threads > rt_h) threads = rt_h;

	rt_stat_hit = rt_stat_sky = rt_stat_miss = 0;
	t0 = Sys_Milliseconds();
	for (i = 0; i < threads; i++) {
		jobs[i].y0 = (rt_h * i) / threads;
		jobs[i].y1 = (rt_h * (i + 1)) / threads;
		jobs[i].seed = rt_frame_seed + (unsigned)i * 7919u + 1u;
		started[i] = (pthread_create(&th[i], NULL, RT_Worker, &jobs[i]) == 0);
		if (!started[i])
			RT_Worker(&jobs[i]);
	}
	for (i = 0; i < threads; i++) {
		if (started[i])
			pthread_join(th[i], NULL);
	}
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
			blend = 0.40f;
		else
			blend = 0.12f;
	}
	rt_first_frame = 0;
	VectorCopy(r_origin, rt_prev_org);
	VectorCopy(viewAxis[0], rt_prev_fwd);

	RT_TonemapBlit(blend);
	RT_DrawQuad();
	RT_WriteDepth();
	RT_DrawEntities();
	{
		static entity_t rt_worldent;
		memset(&rt_worldent, 0, sizeof(rt_worldent));
		VectorCopy(r_newrefdef.vieworg, rt_worldent.origin);
		currententity = &rt_worldent;
	}

	if (rt_dump->integer) {
		static int dump_frames;
		dump_frames++;
		if (dump_frames == 6) {
			RT_DumpTGA();
			if (rt_dump->integer >= 2)
				Cbuf_AddText("quit\n");
		}
	}

	if (r_speeds->integer)
		Com_Printf("rt %ix%i  %i ms  lights %i  thr %i  hit %i sky %i miss %i  org %.0f %.0f %.0f\n",
			rt_w, rt_h, t1 - t0, rt_num_lights, threads,
			rt_stat_hit, rt_stat_sky, rt_stat_miss,
			r_origin[0], r_origin[1], r_origin[2]);

	return true;
}

void R_RayInit(void)
{
	r_raytrace = Cvar_Get("r_raytrace", "1", CVAR_ARCHIVE);
	rt_scale = Cvar_Get("rt_scale", "4", CVAR_ARCHIVE);
	rt_mode = Cvar_Get("rt_mode", "0", CVAR_ARCHIVE);
	rt_bounces = Cvar_Get("rt_bounces", "1", CVAR_ARCHIVE);
	rt_accum = Cvar_Get("rt_accum", "1", CVAR_ARCHIVE);
	rt_exposure = Cvar_Get("rt_exposure", "2.6", CVAR_ARCHIVE);
	rt_lightscale = Cvar_Get("rt_lightscale", "1.35", CVAR_ARCHIVE);
	rt_skylight = Cvar_Get("rt_skylight", "2.4", CVAR_ARCHIVE);
	rt_reflect = Cvar_Get("rt_reflect", "0.42", CVAR_ARCHIVE);
	rt_threads = Cvar_Get("rt_threads", "0", CVAR_ARCHIVE);
	rt_weapons = Cvar_Get("rt_weapons", "1", CVAR_ARCHIVE);
	rt_dump = Cvar_Get("rt_dump", "0", 0);
	rt_tex_ready = false;
	rt_texnum = 0;
	rt_first_frame = 1;
}

void R_RayShutdown(void)
{
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
	rt_buf_size = 0;
	if (rt_tex_ready) {
		qglDeleteTextures(1, &rt_texnum);
		rt_tex_ready = false;
	}
	rt_num_lights = 0;
	rt_num_skylights = 0;
	rt_light_world = NULL;
}
