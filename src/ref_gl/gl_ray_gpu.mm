/*
 * Metal compute path tracer for Quake II (Apple Silicon).
 *
 * Builds a triangle BVH from the BSP, traces on the GPU, and returns
 * RGB (and optional Z) for path-trace blit or luxel lightmap bake.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern "C" {
#include "gl_local.h"
#include "gl_ray_gpu.h"
}

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RTG_MAX_TRIS		200000
#define RTG_MAX_NODES		400000
#define RTG_MAX_LIGHTS		384
#define RTG_LEAF_TRIS		4
#define RTG_MAX_DIST		16384.0f

typedef struct {
	float	v0[4];
	float	v1[4];
	float	v2[4];
	float	n[4];		/* xyz + flags (sky=1, light=2) */
	float	albedo[4];	/* rgb + emit scale */
} rtg_tri_t;

typedef struct {
	float	bmin[4];
	float	bmax[4];
	int		left;		/* child index, or -1 for leaf */
	int		right;		/* child index, or first tri */
	int		count;		/* leaf triangle count */
	int		pad;
} rtg_node_t;

typedef struct {
	float	center[4];
	float	emit[4];	/* rgb + area */
} rtg_light_t;

typedef struct {
	float	origin[4];
	float	forward[4];
	float	left[4];
	float	up[4];
	float	tanx, tany, skylight, lightscale;
	float	skyTint[4];
	float	sunDir[4];
	int		width, height, bounces, numLights;
	unsigned	seed;
	int		numNodes, numTris;
	float	modulate;
	int		spp;
} rtg_uniforms_t;

static id<MTLDevice>			rtg_device;
static id<MTLCommandQueue>		rtg_queue;
static id<MTLLibrary>			rtg_lib;
static id<MTLComputePipelineState> rtg_pipe_path;
static id<MTLComputePipelineState> rtg_pipe_luxel;
static id<MTLBuffer>			rtg_triBuf;
static id<MTLBuffer>			rtg_nodeBuf;
static id<MTLBuffer>			rtg_lightBuf;
static id<MTLBuffer>			rtg_outBuf;
static id<MTLBuffer>			rtg_zBuf;
static id<MTLBuffer>			rtg_uniformBuf;
static id<MTLBuffer>			rtg_luxelInBuf;
static id<MTLBuffer>			rtg_luxelOutBuf;

static rtg_tri_t	*rtg_tris;
static rtg_node_t	*rtg_nodes;
static int			rtg_num_tris;
static int			rtg_num_nodes;
static rtg_light_t	rtg_lights[RTG_MAX_LIGHTS];
static int			rtg_num_lights;
static const void	*rtg_world_key;
static qboolean		rtg_ready;
static size_t		rtg_out_cap;
static size_t		rtg_luxel_cap;

/* ---- embedded Metal Shading Language ---------------------------------- */

static NSString *RTG_ShaderSource(void)
{
	return @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Tri {
	float4 v0;
	float4 v1;
	float4 v2;
	float4 n;      // xyz normal, w flags (1=sky, 2=light)
	float4 albedo; // rgb, w emit
};

struct Node {
	float4 bmin;
	float4 bmax;
	int left;
	int right;
	int count;
	int pad;
};

struct Light {
	float4 center;
	float4 emit; // rgb, w area
};

struct Uniforms {
	float4 origin;
	float4 forward;
	float4 left;
	float4 up;
	float tanx, tany, skylight, lightscale;
	float4 skyTint;
	float4 sunDir;
	int width, height, bounces, numLights;
	uint seed;
	int numNodes, numTris;
	float modulate;
	int spp;
};

static inline uint wang(uint x)
{
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x ? x : 0xA341316Cu;
}

static inline float frand(thread uint &s)
{
	s = wang(s);
	return (s & 0xFFFFFF) / 16777216.0f;
}

static inline float3 skyColor(float3 dir, constant Uniforms &u)
{
	float3 c = u.skyTint.xyz;
	float sun = max(dot(dir, u.sunDir.xyz), 0.0f);
	sun = sun * sun;
	sun = sun * sun;
	sun = sun * sun;
	c += float3(4.5f, 3.6f, 2.4f) * sun;
	return c;
}

static inline bool rayAABB(float3 o, float3 d, float tmin, float tmax, float3 bmin, float3 bmax)
{
	for (int i = 0; i < 3; i++) {
		float inv = 1.0f / (fabs(d[i]) < 1e-8f ? (d[i] >= 0.0f ? 1e-8f : -1e-8f) : d[i]);
		float t0 = (bmin[i] - o[i]) * inv;
		float t1 = (bmax[i] - o[i]) * inv;
		if (inv < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
		tmin = max(tmin, t0);
		tmax = min(tmax, t1);
		if (tmax < tmin)
			return false;
	}
	return true;
}

struct Hit {
	bool hit;
	bool sky;
	float t;
	float3 pos;
	float3 n;
	float3 albedo;
	float emit;
};

static Hit intersectTri(const Tri t, float3 o, float3 d, float tmin, float tmax)
{
	Hit h;
	h.hit = false;
	h.sky = false;
	h.t = tmax;
	h.emit = 0.0f;
	float3 e1 = t.v1.xyz - t.v0.xyz;
	float3 e2 = t.v2.xyz - t.v0.xyz;
	float3 p = cross(d, e2);
	float det = dot(e1, p);
	if (fabs(det) < 1e-8f)
		return h;
	float inv = 1.0f / det;
	float3 s = o - t.v0.xyz;
	float u = dot(s, p) * inv;
	if (u < 0.0f || u > 1.0f)
		return h;
	float3 q = cross(s, e1);
	float v = dot(d, q) * inv;
	if (v < 0.0f || u + v > 1.0f)
		return h;
	float tt = dot(e2, q) * inv;
	if (tt < tmin || tt > tmax)
		return h;
	h.hit = true;
	h.t = tt;
	h.pos = o + d * tt;
	h.n = normalize(t.n.xyz);
	if (dot(h.n, d) > 0.0f)
		h.n = -h.n;
	h.albedo = t.albedo.xyz;
	h.emit = t.albedo.w;
	h.sky = ((int)t.n.w & 1) != 0;
	return h;
}

static Hit trace(float3 o, float3 d, float tmin, float tmax,
	device const Tri *tris, device const Node *nodes, int numNodes)
{
	Hit best;
	best.hit = false;
	best.sky = false;
	best.t = tmax;
	best.emit = 0.0f;
	best.albedo = float3(0.0f);
	best.n = float3(0.0f, 0.0f, 1.0f);
	best.pos = o;

	if (numNodes < 1)
		return best;

	int stack[64];
	int sp = 0;
	stack[sp++] = 0;

	while (sp > 0) {
		int idx = stack[--sp];
		Node node = nodes[idx];
		float nearest = best.hit ? best.t : tmax;
		if (!rayAABB(o, d, tmin, nearest, node.bmin.xyz, node.bmax.xyz))
			continue;
		if (node.left < 0) {
			int first = node.right;
			int count = node.count;
			for (int i = 0; i < count; i++) {
				Hit h = intersectTri(tris[first + i], o, d, tmin, nearest);
				if (h.hit) {
					best = h;
					nearest = h.t;
				}
			}
		} else {
			if (sp < 62) {
				stack[sp++] = node.right;
				stack[sp++] = node.left;
			}
		}
	}
	return best;
}

static bool occluded(float3 o, float3 d, float tmax,
	device const Tri *tris, device const Node *nodes, int numNodes)
{
	Hit h = trace(o, d, 0.05f, tmax - 0.1f, tris, nodes, numNodes);
	return h.hit && !h.sky;
}

static float3 clampFirefly(float3 c, float lim)
{
	float m = max(c.x, max(c.y, c.z));
	if (m > lim)
		c *= lim / m;
	return c;
}

static float3 nee(Hit hit, thread uint &rng, device const Light *lights, int numLights,
	device const Tri *tris, device const Node *nodes, int numNodes)
{
	if (numLights < 1)
		return float3(0.0f);
	int li = (int)(frand(rng) * (float)numLights) % numLights;
	Light L = lights[li];
	float3 toL = L.center.xyz - hit.pos;
	float dist = length(toL);
	if (dist < 2.0f)
		return float3(0.0f);
	toL /= dist;
	float ndotl = dot(hit.n, toL);
	if (ndotl <= 0.0f)
		return float3(0.0f);
	float3 origin = hit.pos + hit.n * 0.15f;
	if (occluded(origin, toL, dist, tris, nodes, numNodes))
		return float3(0.0f);
	float pdf = 1.0f / (float)numLights;
	float att = (ndotl * min(L.emit.w, 4000.0f)) / (dist * dist * pdf);
	att *= 0.35f;
	return clampFirefly(L.emit.xyz * att, 8.0f);
}

static float3 skyNee(Hit hit, thread uint &rng, constant Uniforms &u,
	device const Tri *tris, device const Node *nodes, int numNodes)
{
	float3 outc = float3(0.0f);
	float gain = u.skylight;
	if (gain < 0.01f)
		return outc;
	float3 origin = hit.pos + hit.n * 0.2f;
	/* Stable directions: sun + straight up + a few soft cones */
	float3 dirs[6];
	dirs[0] = u.sunDir.xyz;
	dirs[1] = float3(0.0f, 0.0f, 1.0f);
	dirs[2] = normalize(u.sunDir.xyz + float3(0.08f, 0.0f, 0.0f));
	dirs[3] = normalize(u.sunDir.xyz + float3(-0.06f, 0.05f, 0.0f));
	dirs[4] = normalize(float3(frand(rng) - 0.5f, frand(rng) - 0.5f, 0.85f));
	dirs[5] = normalize(float3(frand(rng) - 0.5f, frand(rng) - 0.5f, 0.7f));
	float wsum = 0.0f;
	for (int i = 0; i < 6; i++) {
		float3 dir = normalize(dirs[i]);
		float ndotl = dot(hit.n, dir);
		if (ndotl <= 0.02f)
			continue;
		float w = (i < 2) ? 2.0f : 1.0f;
		wsum += w * ndotl;
		Hit h = trace(origin, dir, 0.05f, 16384.0f, tris, nodes, u.numNodes);
		if (!h.hit || h.sky)
			outc += skyColor(dir, u) * (ndotl * gain * w * 0.55f);
	}
	(void)wsum;
	return clampFirefly(outc, 12.0f);
}

static float3 shade(Hit hit, float3 rayDir, thread uint &rng, constant Uniforms &u,
	device const Tri *tris, device const Node *nodes, device const Light *lights)
{
	if (hit.sky)
		return skyColor(rayDir, u);

	float3 lighting = float3(0.08f, 0.08f, 0.09f);
	lighting += nee(hit, rng, lights, u.numLights, tris, nodes, u.numNodes);
	lighting += nee(hit, rng, lights, u.numLights, tris, nodes, u.numNodes);
	lighting *= 0.5f;
	lighting += skyNee(hit, rng, u, tris, nodes, u.numNodes);
	if (hit.emit > 0.0f)
		lighting += clampFirefly(hit.albedo * min(hit.emit, 4.0f), 6.0f);

	if (u.bounces > 0) {
		float3 t = fabs(hit.n.z) < 0.9f ? float3(hit.n.y, -hit.n.x, 0.0f) : float3(0.0f, hit.n.z, -hit.n.y);
		t = normalize(t);
		float3 b = cross(hit.n, t);
		float r1 = frand(rng), r2 = frand(rng);
		float r = sqrt(r1);
		float phi = 6.2831853f * r2;
		float3 dir = normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + hit.n * sqrt(1.0f - r1));
		float3 origin = hit.pos + hit.n * 0.15f;
		Hit h2 = trace(origin, dir, 0.05f, 16384.0f, tris, nodes, u.numNodes);
		if (!h2.hit || h2.sky)
			lighting += skyColor(dir, u) * 0.55f;
		else {
			float3 l2 = float3(0.06f) + nee(h2, rng, lights, u.numLights, tris, nodes, u.numNodes) * 0.5f;
			l2 += skyNee(h2, rng, u, tris, nodes, u.numNodes) * 0.45f;
			lighting += clampFirefly(h2.albedo * l2, 4.0f) * 0.55f;
		}
	}
	return clampFirefly(hit.albedo * lighting * u.lightscale, 16.0f);
}

kernel void path_trace(
	device const Tri *tris [[buffer(0)]],
	device const Node *nodes [[buffer(1)]],
	device const Light *lights [[buffer(2)]],
	device float *outRGB [[buffer(3)]],
	device float *outZ [[buffer(4)]],
	constant Uniforms &u [[buffer(5)]],
	uint2 gid [[thread_position_in_grid]])
{
	if ((int)gid.x >= u.width || (int)gid.y >= u.height)
		return;
	uint rng = u.seed ^ (gid.x * 73856093u) ^ (gid.y * 19349663u) ^ 0x9e3779b9u;
	int spp = u.spp;
	if (spp < 1) spp = 1;
	if (spp > 16) spp = 16;

	float3 accum = float3(0.0f);
	float z = 4096.0f;
	float3 origin = u.origin.xyz;
	for (int s = 0; s < spp; s++) {
		float nx = (2.0f * ((float)gid.x + frand(rng)) / (float)u.width - 1.0f) * u.tanx;
		float ny = (1.0f - 2.0f * ((float)gid.y + frand(rng)) / (float)u.height) * u.tany;
		float3 dir = normalize(u.forward.xyz - u.left.xyz * nx + u.up.xyz * ny);
		Hit hit = trace(origin, dir, 0.0f, 16384.0f, tris, nodes, u.numNodes);
		float3 color;
		if (!hit.hit) {
			color = skyColor(dir, u);
		} else {
			if (s == 0) {
				float3 delta = hit.pos - origin;
				float ze = dot(delta, u.forward.xyz);
				if (ze > 4.0f)
					z = ze;
			}
			color = shade(hit, dir, rng, u, tris, nodes, lights);
		}
		accum += color;
	}
	accum /= (float)spp;
	int idx = ((int)gid.y * u.width + (int)gid.x) * 3;
	outRGB[idx + 0] = accum.x;
	outRGB[idx + 1] = accum.y;
	outRGB[idx + 2] = accum.z;
	outZ[(int)gid.y * u.width + (int)gid.x] = z;
}

kernel void bake_luxels(
	device const Tri *tris [[buffer(0)]],
	device const Node *nodes [[buffer(1)]],
	device const Light *lights [[buffer(2)]],
	device const float *posNormal [[buffer(3)]],
	device float *outRGB [[buffer(4)]],
	constant Uniforms &u [[buffer(5)]],
	uint gid [[thread_position_in_grid]])
{
	if ((int)gid >= u.width) /* width == luxel count */
		return;
	uint rng = u.seed ^ (gid * 747796405u) ^ 0x9e3779b9u;
	float3 pos = float3(posNormal[gid * 6 + 0], posNormal[gid * 6 + 1], posNormal[gid * 6 + 2]);
	float3 n = normalize(float3(posNormal[gid * 6 + 3], posNormal[gid * 6 + 4], posNormal[gid * 6 + 5]));
	Hit hit;
	hit.hit = true;
	hit.sky = false;
	hit.t = 0.0f;
	hit.pos = pos;
	hit.n = n;
	hit.albedo = float3(1.0f);
	hit.emit = 0.0f;

	int spp = u.spp;
	if (spp < 1) spp = 1;
	if (spp > 16) spp = 16;
	float3 lighting = float3(0.0f);
	for (int s = 0; s < spp; s++) {
		float3 L = float3(0.06f);
		L += nee(hit, rng, lights, u.numLights, tris, nodes, u.numNodes);
		L += skyNee(hit, rng, u, tris, nodes, u.numNodes);
		lighting += L;
	}
	lighting /= (float)spp;

	float scale = 48.0f * u.modulate * u.lightscale;
	lighting = clampFirefly(lighting * scale, 400.0f);
	outRGB[gid * 3 + 0] = lighting.x;
	outRGB[gid * 3 + 1] = lighting.y;
	outRGB[gid * 3 + 2] = lighting.z;
}
)MSL";
}

/* ---- BVH build -------------------------------------------------------- */

static void RTG_TriBounds(const rtg_tri_t *t, float bmin[3], float bmax[3])
{
	int i;
	for (i = 0; i < 3; i++) {
		bmin[i] = t->v0[i];
		if (t->v1[i] < bmin[i]) bmin[i] = t->v1[i];
		if (t->v2[i] < bmin[i]) bmin[i] = t->v2[i];
		bmax[i] = t->v0[i];
		if (t->v1[i] > bmax[i]) bmax[i] = t->v1[i];
		if (t->v2[i] > bmax[i]) bmax[i] = t->v2[i];
	}
}

static void RTG_Expand(float bmin[3], float bmax[3], const float p[3])
{
	int i;
	for (i = 0; i < 3; i++) {
		if (p[i] < bmin[i]) bmin[i] = p[i];
		if (p[i] > bmax[i]) bmax[i] = p[i];
	}
}

static int RTG_BuildNode(int *order, int start, int count)
{
	int node = rtg_num_nodes++;
	rtg_node_t *n;
	float bmin[3], bmax[3];
	int i, axis, mid;
	float extent[3];

	if (rtg_num_nodes > RTG_MAX_NODES)
		return 0;

	n = &rtg_nodes[node];
	bmin[0] = bmin[1] = bmin[2] = 1e30f;
	bmax[0] = bmax[1] = bmax[2] = -1e30f;
	for (i = 0; i < count; i++) {
		float tbmin[3], tbmax[3];
		RTG_TriBounds(&rtg_tris[order[start + i]], tbmin, tbmax);
		RTG_Expand(bmin, bmax, tbmin);
		RTG_Expand(bmin, bmax, tbmax);
	}
	n->bmin[0] = bmin[0]; n->bmin[1] = bmin[1]; n->bmin[2] = bmin[2]; n->bmin[3] = 0;
	n->bmax[0] = bmax[0]; n->bmax[1] = bmax[1]; n->bmax[2] = bmax[2]; n->bmax[3] = 0;

	if (count <= RTG_LEAF_TRIS) {
		/* pack leaf tris contiguously by swapping into place via order */
		n->left = -1;
		n->right = order[start]; /* will fix after compact - store start in order space */
		n->count = count;
		/* For leaves we need contiguous tris. Compact by rewriting order region
		 * into a contiguous range at the end... Simpler: store indices in a
		 * side array. Easiest fix: make leaf store first index into a remapped
		 * contiguous triangle array built after BVH.
		 * Here: temporarily store start/count into order[]; remapped later.
		 */
		n->right = start; /* index into order[] */
		return node;
	}

	extent[0] = bmax[0] - bmin[0];
	extent[1] = bmax[1] - bmin[1];
	extent[2] = bmax[2] - bmin[2];
	axis = 0;
	if (extent[1] > extent[axis]) axis = 1;
	if (extent[2] > extent[axis]) axis = 2;

	/* partition by centroid */
	{
		int left = start, right = start + count - 1;
		float pivot = 0.5f * (bmin[axis] + bmax[axis]);
		while (left <= right) {
			float c, tbmin[3], tbmax[3];
			RTG_TriBounds(&rtg_tris[order[left]], tbmin, tbmax);
			c = 0.5f * (tbmin[axis] + tbmax[axis]);
			if (c < pivot)
				left++;
			else {
				int tmp = order[left];
				order[left] = order[right];
				order[right] = tmp;
				right--;
			}
		}
		mid = left - start;
		if (mid < 1) mid = 1;
		if (mid >= count) mid = count - 1;
	}

	n->left = RTG_BuildNode(order, start, mid);
	n->right = RTG_BuildNode(order, start + mid, count - mid);
	n->count = 0;
	return node;
}

static qboolean RTG_CompactLeaves(int *order)
{
	/* Rebuild triangle array so each leaf references a contiguous block. */
	rtg_tri_t *packed;
	int *newFirst;
	int i, npacked = 0;

	packed = (rtg_tri_t *)malloc(sizeof(rtg_tri_t) * (size_t)rtg_num_tris);
	newFirst = (int *)malloc(sizeof(int) * (size_t)rtg_num_nodes);
	if (!packed || !newFirst) {
		free(packed);
		free(newFirst);
		return false;
	}

	for (i = 0; i < rtg_num_nodes; i++) {
		rtg_node_t *n = &rtg_nodes[i];
		if (n->left >= 0)
			continue;
		{
			int start = n->right;
			int count = n->count;
			int j;
			newFirst[i] = npacked;
			for (j = 0; j < count; j++)
				packed[npacked++] = rtg_tris[order[start + j]];
			n->right = newFirst[i];
		}
	}
	memcpy(rtg_tris, packed, sizeof(rtg_tri_t) * (size_t)npacked);
	rtg_num_tris = npacked;
	free(packed);
	free(newFirst);
	return true;
}

static void RTG_CollectLights(void)
{
	int i;

	rtg_num_lights = 0;
	if (!r_worldmodel)
		return;
	for (i = 0; i < r_worldmodel->numsurfaces && rtg_num_lights < RTG_MAX_LIGHTS; i++) {
		msurface_t *s = r_worldmodel->surfaces + i;
		glpoly_t *poly;
		float area = 0.0f;
		vec3_t center, emit;
		int n, k;
		const float *v0;

		if (!s->texinfo || !s->polys)
			continue;
		if (!(s->texinfo->flags & SURF_LIGHT) || s->texinfo->value <= 0)
			continue;
		poly = s->polys;
		if (poly->numverts < 3)
			continue;
		v0 = poly->verts[0];
		n = poly->numverts;
		center[0] = center[1] = center[2] = 0.0f;
		for (k = 0; k < n; k++) {
			const float *v = v0 + k * VERTEXSIZE;
			center[0] += v[0]; center[1] += v[1]; center[2] += v[2];
		}
		center[0] /= n; center[1] /= n; center[2] /= n;
		for (k = 1; k + 1 < n; k++) {
			const float *a = v0, *b = v0 + k * VERTEXSIZE, *c = v0 + (k + 1) * VERTEXSIZE;
			vec3_t e1, e2, cr;
			e1[0] = b[0]-a[0]; e1[1] = b[1]-a[1]; e1[2] = b[2]-a[2];
			e2[0] = c[0]-a[0]; e2[1] = c[1]-a[1]; e2[2] = c[2]-a[2];
			cr[0] = e1[1]*e2[2]-e1[2]*e2[1];
			cr[1] = e1[2]*e2[0]-e1[0]*e2[2];
			cr[2] = e1[0]*e2[1]-e1[1]*e2[0];
			area += 0.5f * (float)sqrt(cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]);
		}
		if (area < 1.0f)
			area = 1.0f;
		{
			float scale = s->texinfo->value * 0.008f;
			image_t *img = s->texinfo->image;
			if (img && img->pixels) {
				emit[0] = img->pixels[0] * ONEDIV255 * scale;
				emit[1] = img->pixels[1] * ONEDIV255 * scale;
				emit[2] = img->pixels[2] * ONEDIV255 * scale;
			} else {
				emit[0] = emit[1] = emit[2] = scale;
			}
		}
		rtg_lights[rtg_num_lights].center[0] = center[0];
		rtg_lights[rtg_num_lights].center[1] = center[1];
		rtg_lights[rtg_num_lights].center[2] = center[2];
		rtg_lights[rtg_num_lights].center[3] = 0;
		rtg_lights[rtg_num_lights].emit[0] = emit[0];
		rtg_lights[rtg_num_lights].emit[1] = emit[1];
		rtg_lights[rtg_num_lights].emit[2] = emit[2];
		rtg_lights[rtg_num_lights].emit[3] = area;
		rtg_num_lights++;
	}
}

static void RTG_AddSurfaceTris(msurface_t *surf)
{
	glpoly_t *poly = surf->polys;
	const float *v0;
	vec3_t n, albedo;
	float emit = 0.0f;
	float flags = 0.0f;
	int i, nv;
	image_t *img;

	if (!poly || poly->numverts < 3 || !surf->texinfo)
		return;
	if (surf->texinfo->flags & SURF_NODRAW)
		return;

	if (surf->flags & SURF_PLANEBACK) {
		n[0] = -surf->plane->normal[0];
		n[1] = -surf->plane->normal[1];
		n[2] = -surf->plane->normal[2];
	} else {
		n[0] = surf->plane->normal[0];
		n[1] = surf->plane->normal[1];
		n[2] = surf->plane->normal[2];
	}

	if (surf->texinfo->flags & SURF_SKY)
		flags = 1.0f;
	if (surf->texinfo->flags & SURF_LIGHT) {
		flags += 2.0f;
		emit = surf->texinfo->value * 0.008f;
		if (emit < 1.0f) emit = 1.0f;
	}

	albedo[0] = albedo[1] = albedo[2] = 0.55f;
	img = surf->texinfo->image;
	if (img && img->pixels && img->width > 0 && img->height > 0) {
		int x = img->width / 2, y = img->height / 2;
		const byte *p = img->pixels + (y * img->width + x) * 3;
		albedo[0] = p[0] * ONEDIV255;
		albedo[1] = p[1] * ONEDIV255;
		albedo[2] = p[2] * ONEDIV255;
	}

	v0 = poly->verts[0];
	nv = poly->numverts;
	for (i = 1; i + 1 < nv && rtg_num_tris < RTG_MAX_TRIS; i++) {
		const float *a = v0;
		const float *b = v0 + i * VERTEXSIZE;
		const float *c = v0 + (i + 1) * VERTEXSIZE;
		rtg_tri_t *t = &rtg_tris[rtg_num_tris++];
		t->v0[0] = a[0]; t->v0[1] = a[1]; t->v0[2] = a[2]; t->v0[3] = 0;
		t->v1[0] = b[0]; t->v1[1] = b[1]; t->v1[2] = b[2]; t->v1[3] = 0;
		t->v2[0] = c[0]; t->v2[1] = c[1]; t->v2[2] = c[2]; t->v2[3] = 0;
		t->n[0] = n[0]; t->n[1] = n[1]; t->n[2] = n[2]; t->n[3] = flags;
		t->albedo[0] = albedo[0]; t->albedo[1] = albedo[1]; t->albedo[2] = albedo[2]; t->albedo[3] = emit;
	}
}

static qboolean RTG_BuildWorld(void)
{
	int i, *order;

	if (!r_worldmodel)
		return false;

	if (!rtg_tris)
		rtg_tris = (rtg_tri_t *)malloc(sizeof(rtg_tri_t) * RTG_MAX_TRIS);
	if (!rtg_nodes)
		rtg_nodes = (rtg_node_t *)malloc(sizeof(rtg_node_t) * RTG_MAX_NODES);
	if (!rtg_tris || !rtg_nodes)
		return false;

	rtg_num_tris = 0;
	rtg_num_nodes = 0;
	for (i = 0; i < r_worldmodel->numsurfaces; i++)
		RTG_AddSurfaceTris(r_worldmodel->surfaces + i);

	if (rtg_num_tris < 1)
		return false;

	order = (int *)malloc(sizeof(int) * (size_t)rtg_num_tris);
	if (!order)
		return false;
	for (i = 0; i < rtg_num_tris; i++)
		order[i] = i;

	RTG_BuildNode(order, 0, rtg_num_tris);
	if (!RTG_CompactLeaves(order)) {
		free(order);
		return false;
	}
	free(order);

	RTG_CollectLights();

	/* Upload GPU buffers */
	rtg_triBuf = [rtg_device newBufferWithBytes:rtg_tris
										length:sizeof(rtg_tri_t) * (size_t)rtg_num_tris
									   options:MTLResourceStorageModeShared];
	rtg_nodeBuf = [rtg_device newBufferWithBytes:rtg_nodes
										 length:sizeof(rtg_node_t) * (size_t)rtg_num_nodes
										options:MTLResourceStorageModeShared];
	rtg_lightBuf = [rtg_device newBufferWithBytes:rtg_lights
										  length:sizeof(rtg_light_t) * (size_t)(rtg_num_lights > 0 ? rtg_num_lights : 1)
										 options:MTLResourceStorageModeShared];
	if (!rtg_triBuf || !rtg_nodeBuf || !rtg_lightBuf)
		return false;

	rtg_world_key = r_worldmodel;
	Com_Printf("raytracer: GPU BVH %i tris, %i nodes, %i lights\n",
		rtg_num_tris, rtg_num_nodes, rtg_num_lights);
	return true;
}

static void RTG_FillSkyUniforms(rtg_uniforms_t *u)
{
	vec3_t up = {0, 0, 1}, sun = {0.52f, 0.28f, 0.81f}, sky;
	float len;

	len = (float)sqrt(sun[0]*sun[0]+sun[1]*sun[1]+sun[2]*sun[2]);
	if (len > 1e-4f) { sun[0]/=len; sun[1]/=len; sun[2]/=len; }
	R_SampleSkyColor(up, sky);
	u->skyTint[0] = sky[0] * 1.35f + 0.08f;
	u->skyTint[1] = sky[1] * 1.25f + 0.06f;
	u->skyTint[2] = sky[2] * 1.15f + 0.05f;
	u->skyTint[3] = 0;
	u->sunDir[0] = sun[0];
	u->sunDir[1] = sun[1];
	u->sunDir[2] = sun[2];
	u->sunDir[3] = 0;
}

/* ---- public API ------------------------------------------------------- */

extern "C" qboolean RT_GPU_Init(void)
{
	NSError *err = nil;
	id<MTLFunction> fPath, fLuxel;

	rtg_ready = false;
	rtg_device = MTLCreateSystemDefaultDevice();
	if (!rtg_device) {
		Com_Printf("raytracer: no Metal device\n");
		return false;
	}
	rtg_queue = [rtg_device newCommandQueue];
	if (!rtg_queue)
		return false;

	rtg_lib = [rtg_device newLibraryWithSource:RTG_ShaderSource() options:nil error:&err];
	if (!rtg_lib) {
		Com_Printf("raytracer: Metal library failed: %s\n",
			err ? [[err localizedDescription] UTF8String] : "?");
		return false;
	}
	fPath = [rtg_lib newFunctionWithName:@"path_trace"];
	fLuxel = [rtg_lib newFunctionWithName:@"bake_luxels"];
	if (!fPath || !fLuxel)
		return false;
	rtg_pipe_path = [rtg_device newComputePipelineStateWithFunction:fPath error:&err];
	rtg_pipe_luxel = [rtg_device newComputePipelineStateWithFunction:fLuxel error:&err];
	if (!rtg_pipe_path || !rtg_pipe_luxel) {
		Com_Printf("raytracer: Metal PSO failed: %s\n",
			err ? [[err localizedDescription] UTF8String] : "?");
		return false;
	}
	rtg_uniformBuf = [rtg_device newBufferWithLength:sizeof(rtg_uniforms_t)
											 options:MTLResourceStorageModeShared];
	if (!rtg_uniformBuf)
		return false;

	rtg_ready = true;
	Com_Printf("raytracer: Metal GPU path tracer ready (%s)\n",
		[[rtg_device name] UTF8String]);
	return true;
}

extern "C" void RT_GPU_Shutdown(void)
{
	rtg_triBuf = nil;
	rtg_nodeBuf = nil;
	rtg_lightBuf = nil;
	rtg_outBuf = nil;
	rtg_zBuf = nil;
	rtg_uniformBuf = nil;
	rtg_luxelInBuf = nil;
	rtg_luxelOutBuf = nil;
	rtg_pipe_path = nil;
	rtg_pipe_luxel = nil;
	rtg_lib = nil;
	rtg_queue = nil;
	rtg_device = nil;
	free(rtg_tris); rtg_tris = NULL;
	free(rtg_nodes); rtg_nodes = NULL;
	rtg_num_tris = rtg_num_nodes = 0;
	rtg_world_key = NULL;
	rtg_ready = false;
	rtg_out_cap = rtg_luxel_cap = 0;
}

extern "C" qboolean RT_GPU_Active(void)
{
	return rtg_ready;
}

extern "C" void RT_GPU_InvalidateWorld(void)
{
	rtg_world_key = NULL;
	rtg_triBuf = nil;
	rtg_nodeBuf = nil;
	rtg_lightBuf = nil;
	rtg_num_tris = 0;
	rtg_num_nodes = 0;
}

extern "C" qboolean RT_GPU_EnsureWorld(void)
{
	if (!rtg_ready)
		return false;
	if (rtg_world_key == r_worldmodel && rtg_triBuf && rtg_nodeBuf)
		return true;
	return RTG_BuildWorld();
}

extern "C" qboolean RT_GPU_PathTrace(int w, int h, float tanx, float tany,
	const float origin[3], const float forward[3], const float left[3], const float up[3],
	float skylight, float lightscale, int bounces, int spp, unsigned seed,
	float *outRGB, float *outZ)
{
	rtg_uniforms_t *u;
	id<MTLCommandBuffer> cmd;
	id<MTLComputeCommandEncoder> enc;
	MTLSize grid, tg;
	size_t need;
	NSUInteger tgSize;

	if (!rtg_ready || !outRGB || w < 1 || h < 1)
		return false;
	if (!RT_GPU_EnsureWorld())
		return false;

	need = (size_t)w * (size_t)h * 3 * sizeof(float);
	if (!rtg_outBuf || rtg_out_cap < need) {
		rtg_outBuf = [rtg_device newBufferWithLength:need options:MTLResourceStorageModeShared];
		rtg_zBuf = [rtg_device newBufferWithLength:(size_t)w * h * sizeof(float)
										   options:MTLResourceStorageModeShared];
		rtg_out_cap = need;
		if (!rtg_outBuf || !rtg_zBuf)
			return false;
	}

	if (spp < 1) spp = 1;
	if (spp > 16) spp = 16;

	u = (rtg_uniforms_t *)[rtg_uniformBuf contents];
	memset(u, 0, sizeof(*u));
	u->origin[0] = origin[0]; u->origin[1] = origin[1]; u->origin[2] = origin[2];
	u->forward[0] = forward[0]; u->forward[1] = forward[1]; u->forward[2] = forward[2];
	u->left[0] = left[0]; u->left[1] = left[1]; u->left[2] = left[2];
	u->up[0] = up[0]; u->up[1] = up[1]; u->up[2] = up[2];
	u->tanx = tanx; u->tany = tany;
	u->skylight = skylight; u->lightscale = lightscale;
	u->width = w; u->height = h;
	u->bounces = bounces > 0 ? 1 : 0;
	u->numLights = rtg_num_lights;
	u->seed = seed;
	u->numNodes = rtg_num_nodes;
	u->numTris = rtg_num_tris;
	u->modulate = 1.0f;
	u->spp = spp;
	RTG_FillSkyUniforms(u);

	cmd = [rtg_queue commandBuffer];
	enc = [cmd computeCommandEncoder];
	[enc setComputePipelineState:rtg_pipe_path];
	[enc setBuffer:rtg_triBuf offset:0 atIndex:0];
	[enc setBuffer:rtg_nodeBuf offset:0 atIndex:1];
	[enc setBuffer:rtg_lightBuf offset:0 atIndex:2];
	[enc setBuffer:rtg_outBuf offset:0 atIndex:3];
	[enc setBuffer:rtg_zBuf offset:0 atIndex:4];
	[enc setBuffer:rtg_uniformBuf offset:0 atIndex:5];

	tgSize = rtg_pipe_path.maxTotalThreadsPerThreadgroup;
	if (tgSize > 256) tgSize = 256;
	tg = MTLSizeMake(16, tgSize / 16, 1);
	if (tg.height < 1) tg.height = 1;
	grid = MTLSizeMake((NSUInteger)w, (NSUInteger)h, 1);
	[enc dispatchThreads:grid threadsPerThreadgroup:tg];
	[enc endEncoding];
	[cmd commit];
	[cmd waitUntilCompleted];

	memcpy(outRGB, [rtg_outBuf contents], need);
	if (outZ)
		memcpy(outZ, [rtg_zBuf contents], (size_t)w * h * sizeof(float));
	return true;
}

extern "C" qboolean RT_GPU_BakeLuxels(int count, const float *posNormal,
	float skylight, float lightscale, float modulate, int spp, unsigned seed,
	float *outRGB)
{
	rtg_uniforms_t *u;
	id<MTLCommandBuffer> cmd;
	id<MTLComputeCommandEncoder> enc;
	MTLSize grid, tg;
	size_t inBytes, outBytes;
	NSUInteger tgSize;

	if (!rtg_ready || count < 1 || !posNormal || !outRGB)
		return false;
	if (!RT_GPU_EnsureWorld())
		return false;

	inBytes = (size_t)count * 6 * sizeof(float);
	outBytes = (size_t)count * 3 * sizeof(float);
	if (!rtg_luxelInBuf || rtg_luxel_cap < inBytes) {
		rtg_luxelInBuf = [rtg_device newBufferWithLength:inBytes options:MTLResourceStorageModeShared];
		rtg_luxelOutBuf = [rtg_device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
		rtg_luxel_cap = inBytes;
		if (!rtg_luxelInBuf || !rtg_luxelOutBuf)
			return false;
	}

	memcpy([rtg_luxelInBuf contents], posNormal, inBytes);

	if (spp < 2) spp = 2;
	if (spp > 16) spp = 16;

	u = (rtg_uniforms_t *)[rtg_uniformBuf contents];
	memset(u, 0, sizeof(*u));
	u->skylight = skylight;
	u->lightscale = lightscale;
	u->modulate = modulate;
	u->width = count; /* luxel count */
	u->height = 1;
	u->bounces = 0;
	u->numLights = rtg_num_lights;
	u->seed = seed;
	u->numNodes = rtg_num_nodes;
	u->numTris = rtg_num_tris;
	u->spp = spp;
	RTG_FillSkyUniforms(u);
	u->sunDir[0] = 0.52f; u->sunDir[1] = 0.28f; u->sunDir[2] = 0.81f;

	cmd = [rtg_queue commandBuffer];
	enc = [cmd computeCommandEncoder];
	[enc setComputePipelineState:rtg_pipe_luxel];
	[enc setBuffer:rtg_triBuf offset:0 atIndex:0];
	[enc setBuffer:rtg_nodeBuf offset:0 atIndex:1];
	[enc setBuffer:rtg_lightBuf offset:0 atIndex:2];
	[enc setBuffer:rtg_luxelInBuf offset:0 atIndex:3];
	[enc setBuffer:rtg_luxelOutBuf offset:0 atIndex:4];
	[enc setBuffer:rtg_uniformBuf offset:0 atIndex:5];

	tgSize = rtg_pipe_luxel.maxTotalThreadsPerThreadgroup;
	if (tgSize > 256) tgSize = 256;
	tg = MTLSizeMake(tgSize, 1, 1);
	grid = MTLSizeMake((NSUInteger)count, 1, 1);
	[enc dispatchThreads:grid threadsPerThreadgroup:tg];
	[enc endEncoding];
	[cmd commit];
	[cmd waitUntilCompleted];

	memcpy(outRGB, [rtg_luxelOutBuf contents], outBytes);
	return true;
}
