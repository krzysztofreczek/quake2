/*
 * Metal GPU path tracer (macOS / Apple Silicon).
 * Falls back to the CPU tracer in gl_ray.c when unavailable.
 */
#ifndef GL_RAY_GPU_H
#define GL_RAY_GPU_H

#include "../game/q_shared.h"

qboolean	RT_GPU_Init(void);
void		RT_GPU_Shutdown(void);
qboolean	RT_GPU_Active(void);
void		RT_GPU_InvalidateWorld(void);

/* Rebuild triangle BVH from r_worldmodel when needed. */
qboolean	RT_GPU_EnsureWorld(void);

/*
 * Fullscreen path-trace into tightly packed RGB floats (w*h*3) and optional
 * eye-space Z (w*h). Returns false to fall back to CPU.
 */
qboolean	RT_GPU_PathTrace(int w, int h, float tanx, float tany,
	const float origin[3], const float forward[3], const float left[3], const float up[3],
	float skylight, float lightscale, int bounces, int spp, unsigned seed,
	float *outRGB, float *outZ);

/*
 * Bake lighting for luxels. posNormal is count*6 (pos.xyz, n.xyz).
 * outRGB is count*3 in lightmap blocklight scale (pre-modulate style).
 */
qboolean	RT_GPU_BakeLuxels(int count, const float *posNormal,
	float skylight, float lightscale, float modulate, int spp, unsigned seed,
	float *outRGB);

#endif
