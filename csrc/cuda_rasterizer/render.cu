#include "../ops.h"

namespace flashgs {
namespace {


__device__ const float near_n = 0.2;
__device__ const float far_n = 100.0;
__device__ const float FilterSize = 0.707106; // sqrt(2) / 2
__device__ const float FilterInvSquare = 2.0f;

__forceinline__ __device__ float3 cross(float3 a, float3 b){return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}

__forceinline__ __device__ float3 operator*(float3 a, float3 b){return make_float3(a.x * b.x, a.y * b.y, a.z*b.z);}

__forceinline__ __device__ float2 operator*(float2 a, float2 b){return make_float2(a.x * b.x, a.y * b.y);}

__forceinline__ __device__ float3 operator*(float f, float3 a){return make_float3(f * a.x, f * a.y, f * a.z);}

__forceinline__ __device__ float2 operator*(float f, float2 a){return make_float2(f * a.x, f * a.y);}

__forceinline__ __device__ float3 operator-(float3 a, float3 b){return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}

__forceinline__ __device__ float2 operator-(float2 a, float2 b){return make_float2(a.x - b.x, a.y - b.y);}

__forceinline__ __device__ float sumf3(float3 a){return a.x + a.y + a.z;}

__forceinline__ __device__ float sumf2(float2 a){return a.x + a.y;}

__forceinline__ __device__ float3 sqrtf3(float3 a){return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));}

__forceinline__ __device__ float2 sqrtf2(float2 a){return make_float2(sqrtf(a.x), sqrtf(a.y));}

__forceinline__ __device__ float3 minf3(float f, float3 a){return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}

__forceinline__ __device__ float2 minf2(float f, float2 a){return make_float2(min(f, a.x), min(f, a.y));}

__forceinline__ __device__ float3 maxf3(float f, float3 a){return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}

__forceinline__ __device__ float2 maxf2(float f, float2 a){return make_float2(max(f, a.x), max(f, a.y));}


// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, int2* ranges)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32; //32
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; //32
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__forceinline__ __device__ float fast_ex2_ftz_f32(float x)
{
	float y;
	asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
	return y;
}

__forceinline__ __device__ void pixel_shader(float3& C, float2& O,
float& T, float& D, float& M1, float& M2, float& distortion, float& median_depth,
float3& N, float2 pixf, float2 xy, float4 con_o, float4 nor_o, float3 rgb, float2 other_feat, float3 Tu, float3 Tv, float3 Tw)
{
	// 3DGS calculation
		// float2 d = { xy.x - pixf.x, xy.y - pixf.y };
		// //float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
		// float power = con_o.w + con_o.x * d.x * d.x + con_o.z * d.y * d.y + con_o.y * d.x * d.y;
		// float alpha;
		// asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(alpha) : "f"(power));
		// //alpha = min(0.99f, alpha);

	// 2DGS calculation
	float3 k = pixf.x * Tw - Tu;
	float3 l = pixf.y * Tw - Tv;
	float3 p = cross(k, l);
	// if (abs(p.z) < 1e-6) { // 如果p.z太小，则此像素的alpha为0
	// 	// T, C, N 等不需要修改，因为alpha会是0
	// 	return; 
	// }
	float2 s = {p.x / p.z, p.y / p.z};
	float rho3d = (s.x * s.x + s.y * s.y); 
	float2 d = {xy.x - pixf.x, xy.y - pixf.y};
	float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
	// compute intersection and depth
	float rho = min(rho3d, rho2d);
	float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
	// if (depth < near_n) continue;
	float opa = nor_o.w;
	float power = -0.5f * rho;
	float alpha = min(0.99f, opa * exp(power));

	float w = alpha * T;
	C.x += rgb.x * w;
	C.y += rgb.y * w;
	C.z += rgb.z * w;
	T -= alpha * T;

	O.x += other_feat.x * w;
	O.y += other_feat.y * w;

	// Render depth distortion map
	// Efficient implementation of distortion loss, see 2DGS' paper appendix.
	float A = 1-T;	
	float m = far_n / (far_n - near_n) * (1 - near_n / depth);
	distortion += (m * m * A + M2 - 2 * m * M1) * w;
	M1 += m * w;
	M2 += m * m * w;

	if (T > 0.5) {
		median_depth = depth;
	}

	D  += depth * w;

	// Render normal map
	N.x += nor_o.x * w;
	N.y += nor_o.y * w;
	N.z += nor_o.z * w;
}

__forceinline__ __device__ uint8_t encode(float x)
{
	return (uint8_t)min(max(0.0f, x * 255.0f), 255.0f);
}

__forceinline__ __device__ uint8_t write_output(float3* __restrict__ out_color, float2* __restrict__ out_other_features,
float* __restrict__ out_depth,	float* __restrict__ out_alpha,
float3* __restrict__ out_normal, float* __restrict__ out_median_depth,
float* __restrict__ out_distortion,
float3 bg_color, int2 pix, int width, int height,
float3 C, float2 O, float T, float D, float3 N, float distortion, float median_depth)
{
	if (pix.x < width && pix.y < height)
	{
		int pix_id = width * pix.y + pix.x;
		if (T < 0.0001f)
		{
			T = 0.0f;
		}
		// out_color[pix_id].x = encode(C.x + T * bg_color.x);
		// out_color[pix_id].y = encode(C.y + T * bg_color.y);
		// out_color[pix_id].z = encode(C.z + T * bg_color.z);

		// out_normal[pix_id].x = encode(N.x);
		// out_normal[pix_id].y = encode(N.y);
		// out_normal[pix_id].z = encode(N.z);

		out_color[pix_id].x = C.x + T * bg_color.x;
		out_color[pix_id].y = C.y + T * bg_color.y;
		out_color[pix_id].z = C.z + T * bg_color.z;

		out_other_features[pix_id].x = O.x + T * bg_color.x;
		out_other_features[pix_id].y = O.y + T * bg_color.y;

		out_depth[pix_id] = D;
		out_alpha[pix_id] = 1 - T;

		out_normal[pix_id].x = N.x;
		out_normal[pix_id].y = N.y;
		out_normal[pix_id].z = N.z;

		out_median_depth[pix_id] = median_depth;
		out_distortion[pix_id] = distortion;

	}
}

struct render_load_info
{
	const void* data[FLASHGS_WARP_SIZE] = { nullptr };
	int lg2_scale[FLASHGS_WARP_SIZE] = { 0 };
	int lg2_scale_T[FLASHGS_WARP_SIZE] = { 0 };

	render_load_info(const uint32_t* point_list, const float2* points_xy, const float3* rgb, const float4* conic_opacity, const float4* normal_opacity, const float2* other_features, const float3* Tu, const float3* Tv, const float3* Tw)
	{
		for (int lane = 0; lane < 32; lane++)
		{
			switch (lane)
			{
			case 0:
				data[lane] = point_list;
				lg2_scale[lane] = 2;
				break;
			case 1:
				data[lane] = &points_xy->x;
				lg2_scale[lane] = 3;
				break;
			case 2:
				data[lane] = &points_xy->y;
				lg2_scale[lane] = 3;
				break;
			case 3:
				data[lane] = &rgb->x;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 4:
				data[lane] = &rgb->y;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 5:
				data[lane] = &rgb->z;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 6:
				data[lane] = &conic_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 7:
				data[lane] = &conic_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 8:
				data[lane] = &conic_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 9:
				data[lane] = &conic_opacity->w;
				lg2_scale[lane] = 4;
				break;
			case 10:
				data[lane] = &normal_opacity->x;
				lg2_scale[lane] = 4;
				break;
			case 11:
				data[lane] = &normal_opacity->y;
				lg2_scale[lane] = 4;
				break;
			case 12:
				data[lane] = &normal_opacity->z;
				lg2_scale[lane] = 4;
				break;
			case 13:
				data[lane] = &normal_opacity->w;
				lg2_scale[lane] = 4;
				break;
			case 14:
				data[lane] = &other_features->x;
				lg2_scale[lane] = 3;
				break;
			case 15:
				data[lane] = &other_features->y;
				lg2_scale[lane] = 3;
				break;
			case 16:
				data[lane] = &Tu->x;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 17:
				data[lane] = &Tu->y;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 18:
				data[lane] = &Tu->z;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 19:
				data[lane] = &Tv->x;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 20:
				data[lane] = &Tv->y;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 21:
				data[lane] = &Tv->z;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 22:
				data[lane] = &Tw->x;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 23:
				data[lane] = &Tw->y;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			case 24:
				data[lane] = &Tw->z;
				lg2_scale[lane] = 3;
				lg2_scale_T[lane] = 2;
				break;
			}
		}
	}
};

__forceinline__ __device__ void get_gaussian_features(float2& xy, float3& rgb, float4& con_o, float4& nor_o, float2& other_feat, float3& Tu, float3& Tv, float3& Tw, float buf, int offset)
{
	xy = {
		__shfl_sync(~0, buf, 1 + offset),
		__shfl_sync(~0, buf, 2 + offset)
	};
	rgb = {
		__shfl_sync(~0, buf, 3 + offset),
		__shfl_sync(~0, buf, 4 + offset),
		__shfl_sync(~0, buf, 5 + offset)
	};
	con_o = {
		__shfl_sync(~0, buf, 6 + offset),
		__shfl_sync(~0, buf, 7 + offset),
		__shfl_sync(~0, buf, 8 + offset),
		__shfl_sync(~0, buf, 9 + offset)
	};
	nor_o = {
		__shfl_sync(~0, buf, 10 + offset),
		__shfl_sync(~0, buf, 11 + offset),
		__shfl_sync(~0, buf, 12 + offset),
		__shfl_sync(~0, buf, 13 + offset)
	};
	other_feat = {
		__shfl_sync(~0, buf, 14 + offset),
		__shfl_sync(~0, buf, 15 + offset)
	};
	Tu = {
		__shfl_sync(~0, buf, 16 + offset),
		__shfl_sync(~0, buf, 17 + offset),
		__shfl_sync(~0, buf, 18 + offset)
	};
	Tv = {
		__shfl_sync(~0, buf, 19 + offset),
		__shfl_sync(~0, buf, 20 + offset),
		__shfl_sync(~0, buf, 21 + offset)
	};
	Tw = {
		__shfl_sync(~0, buf, 22 + offset),
		__shfl_sync(~0, buf, 23 + offset),
		__shfl_sync(~0, buf, 24 + offset)
	};
}

template<int BLOCK_X, int BLOCK_Y, int THREAD_X, int THREAD_Y>
__global__ void renderCUDA(
	const int2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int width, int height, int x_blocks,
	const float2* __restrict__ points_xy,
	const float3* __restrict__ rgb,
	const float2* __restrict__ other_features,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ normal_opacity,
	render_load_info info,
	float3 bg_color,
	float3* __restrict__ out_color,
	float2* __restrict__ out_other_features,
	float* __restrict__ out_depth,
	float* __restrict__ out_alpha,
	float3* __restrict__ out_normal,
	float* __restrict__ out_median_depth,
	float* __restrict__ out_distortion)
{
	int2 range = ranges[blockIdx.y * x_blocks + blockIdx.x];
	int lane = threadIdx.y * blockDim.x + threadIdx.x;
	const void* data = info.data[lane];
	int lg2_scale = info.lg2_scale[lane];
	int lg2_scale_T = info.lg2_scale_T[lane];

	// uint2 pix = { blockIdx.x * BLOCK_X + threadIdx.x, blockIdx.y * BLOCK_Y + threadIdx.y };
	int2 pix[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pix[i][j] = {
				(int)blockIdx.x * BLOCK_X + (int)threadIdx.x * THREAD_X + j,
				(int)blockIdx.y * BLOCK_Y + (int)threadIdx.y * THREAD_Y + i
			};
		}
	}

	// float2 pixf = { (float)pix.x, (float)pix.y };
	float2 pixf[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			pixf[i][j] = { (float)pix[i][j].x, (float)pix[i][j].y };
		}
	}

	float T[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			T[i][j] = 1.0f;
		}
	}

	float3 C[THREAD_Y][THREAD_X];
	float2 O[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			C[i][j] = { 0.0f, 0.0f, 0.0f };
			O[i][j] = { 0.0f, 0.0f };
		}
	}

	float3 N[THREAD_Y][THREAD_X];
	float D[THREAD_Y][THREAD_X];
	float M1[THREAD_Y][THREAD_X];
	float M2[THREAD_Y][THREAD_X];
	float distortion[THREAD_Y][THREAD_X];
	float median_depth[THREAD_Y][THREAD_X];
	float median_contributor[THREAD_Y][THREAD_X];
#pragma unroll
	for (int i = 0; i < THREAD_Y; i++)
	{
#pragma unroll
		for (int j = 0; j < THREAD_X; j++)
		{
			N[i][j] = { 0.0f, 0.0f, 0.0f };
			D[i][j] = 0.0f;
			M1[i][j] = 0.0f;
			M2[i][j] = 0.0f;
			distortion[i][j] = 0.0f;
			median_depth[i][j] = 0.0f;
			median_contributor[i][j] = -1.0f;
		}
	}

	int point_id = range.x;
	if (point_id < range.y)
	{
		int offset;
		float2 xy;
		float3 rgb;
		float4 con_o;
		float4 nor_o;
		float2 other_feat;
		float3 Tu;
		float3 Tv;
		float3 Tw;
		if (lane == 0)
		{
			offset = point_id + 1;
		}
		else if (point_id + 1 < range.y)
		{
			offset = point_list[point_id + 0];
		}
		const float* ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
		if (lg2_scale_T){
			ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale) + ((int64_t)offset << lg2_scale_T));	
		}
		float buf;
		bool load_enable = data != nullptr;
		if (lane == 0)
		{
			load_enable = load_enable && point_id + 1 < range.y;
		}
		else
		{
			load_enable = load_enable && point_id + 0 < range.y;
		}
		if (load_enable)
		{
			buf = __ldg(ptr); // 0: point_list[point_id + 2], 4: point_list[point_id + 3], 8: features[point_list[point_id + 0]], 12: features[point_list[point_id + 1]]
		}

		load_enable = data != nullptr;

		bool done = false;
		while (__any_sync(~0, point_id + 2 < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), 0);
			if (lane == 0)
			{
				offset = point_id + 2;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
			if (lg2_scale_T){
				ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale) + ((int64_t)offset << lg2_scale_T));	
			}
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, nor_o, other_feat, Tu, Tv, Tw, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], O[i][j],
						T[i][j], D[i][j], M1[i][j], M2[i][j], distortion[i][j], median_depth[i][j],
						N[i][j], pixf[i][j], xy, con_o, nor_o, rgb, other_feat, Tu, Tv, Tw);
				}
			}

// 			get_gaussian_features(xy, rgb, con_o, buf, 4);
// #ifdef _DEBUG
// 			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
// 			{
// 				printf("error6\n");
// 			}
// #endif

// 	#pragma unroll
// 			for (int i = 0; i < THREAD_Y; i++)
// 			{
// 	#pragma unroll
// 				for (int j = 0; j < THREAD_X; j++)
// 				{
// 					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
// 				}
// 			}

			done = true;
	#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
	#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}

			point_id += 1;
			buf = ldg_buf;
		}
		while (__any_sync(~0, point_id < range.y && !done))
		{
			offset = __shfl_sync(~0, __float_as_uint(buf), lane & 0);
			if (lane == 0)
			{
				offset = point_id + 2;
			}

			if (lane == 0)
			{
				load_enable = load_enable && point_id + 2 < range.y;
			}
			else
			{
				load_enable = load_enable && point_id + 1 < range.y;
			}

#ifdef _DEBUG
			if (lane == 0)
			{
				printf("point_id = %d\n", point_id);
			}
#endif
			float ldg_buf;
			ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale));
			if (lg2_scale_T){
				ptr = reinterpret_cast<const float*>(reinterpret_cast<const char*>(data) + ((int64_t)offset << lg2_scale) + ((int64_t)offset << lg2_scale_T));	
			}
			if (load_enable)
			{
				ldg_buf = __ldg(ptr); // 0: point_list[point_id + 4], 4: point_list[point_id + 5], 8: features[point_list[point_id + 2]], 12: features[point_list[point_id + 3]]
#ifdef _DEBUG
				if (lane == 0 && __float_as_int(ldg_buf) != point_list[point_id + 4])
				{
					printf("error1\n");
				}
				else if (lane == 4 && __float_as_int(ldg_buf) != point_list[point_id + 5])
				{
					printf("error2\n");
				}
				else if (lane == 8 && ldg_buf != points_xy[point_list[point_id + 2]].x)
				{
					printf("error3\n");
				}
				else if (lane == 12 && ldg_buf != points_xy[point_list[point_id + 3]].x)
				{
					printf("error4\n");
				}
#endif
			}

			get_gaussian_features(xy, rgb, con_o, nor_o, other_feat, Tu, Tv, Tw, buf, 0);
#ifdef _DEBUG
			if (lane == 3 && xy.x != points_xy[point_list[point_id + 0]].x)
			{
				printf("error5\n");
			}
#endif

#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					pixel_shader(C[i][j], O[i][j],
						T[i][j], D[i][j], M1[i][j], M2[i][j], distortion[i][j], median_depth[i][j],
						N[i][j], pixf[i][j], xy, con_o, nor_o, rgb, other_feat, Tu, Tv, Tw);
				}
			}

			if (point_id + 1 >= range.y)
				break;

// 			get_gaussian_features(xy, rgb, con_o, buf, 4);
// #ifdef _DEBUG
// 			if (lane == 3 && xy.x != points_xy[point_list[point_id + 1]].x)
// 			{
// 				printf("error6\n");
// 			}
// #endif

// #pragma unroll
// 			for (int i = 0; i < THREAD_Y; i++)
// 			{
// #pragma unroll
// 				for (int j = 0; j < THREAD_X; j++)
// 				{
// 					pixel_shader(C[i][j], T[i][j], pixf[i][j], xy, con_o, rgb);
// 				}
// 			}

			done = true;
#pragma unroll
			for (int i = 0; i < THREAD_Y; i++)
			{
#pragma unroll
				for (int j = 0; j < THREAD_X; j++)
				{
					done = done && T[i][j] < 0.0001f;
				}
			}
			point_id += 1;
			buf = ldg_buf;
		}
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				write_output(out_color, out_other_features, out_depth, out_alpha, out_normal, out_median_depth,
					out_distortion, bg_color, pix[i][j], width, height,
					C[i][j], O[i][j], T[i][j], D[i][j], N[i][j], distortion[i][j], median_depth[i][j]);
			}
		}
	}
	else
	{
#pragma unroll
		for (int i = 0; i < THREAD_Y; i++)
		{
#pragma unroll
			for (int j = 0; j < THREAD_X; j++)
			{
				int pix_x = blockIdx.x * BLOCK_X + threadIdx.x * THREAD_X + j;
				int pix_y = blockIdx.y * BLOCK_Y + threadIdx.y * THREAD_Y + i;
				int pix_id = width * pix_y + pix_x;
				out_color[pix_id].x = encode(bg_color.x);
				out_color[pix_id].y = encode(bg_color.y);
				out_color[pix_id].z = encode(bg_color.z);
			}
		}
	}
}

template<int BLOCK_X, int BLOCK_Y>
void render(int num_rendered,
	int width, int height,
	float2* points_xy, float3* rgb, float2* other_features,
	float4* conic_opacity, float4* normal_opacity,
	float3* Tu, float3* Tv, float3* Tw,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float3* out_color, float2* out_other_features,
	float* out_depth, float* out_alpha, float3* out_normal,
	float* out_median_depth, float* out_distortion, cudaStream_t stream)
{
	dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	cudaMemsetAsync(ranges, 0, sizeof(int2) * grid.x * grid.y, stream);

    // Identify start and end of per-tile workloads in sorted list
    identifyTileRanges<<<(num_rendered + 255) / 256, 256, 0, stream>>>(
        num_rendered,
        gaussian_keys_sorted,
        ranges);

    // Let each tile blend its range of Gaussians independently in parallel
    renderCUDA<BLOCK_X, BLOCK_Y, BLOCK_X / 8, BLOCK_Y / 4><<<grid, dim3(8, 4, 1), 0, stream>>>(
        ranges,
        gaussian_values_sorted,
        width, height, grid.x,
        points_xy,
        rgb,
		other_features,
        conic_opacity,
		normal_opacity,
        render_load_info(gaussian_values_sorted, points_xy, rgb, conic_opacity, normal_opacity, other_features, Tu, Tv, Tw),
        bg_color,
        out_color,
		out_other_features,
		out_depth,
		out_alpha,
		out_normal,
		out_median_depth,
		out_distortion);
}

} // namespace

void render_16x16(int num_rendered,
	int width, int height,
	float2* points_xy, float3* rgb, float2* other_features,
	float4* conic_opacity, float4* normal_opacity,
	float3* Tu, float3* Tv, float3* Tw,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float3* out_color, float2* out_other_features,
	float* out_depth, float* out_alpha, float3* out_normal,
	float* out_median_depth, float* out_distortion, cudaStream_t stream)
{
    render<16, 16>(num_rendered, width, height, points_xy, rgb, other_features,
		conic_opacity, normal_opacity, Tu, Tv, Tw,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, out_other_features,
		out_depth, out_alpha, out_normal, out_median_depth, out_distortion, stream);
}

void render_32x16(int num_rendered,
	int width, int height,
	float2* points_xy, float3* rgb, float2* other_features,
	float4* conic_opacity, float4* normal_opacity,
	float3* Tu, float3* Tv, float3* Tw,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float3* out_color, float2* out_other_features,
	float* out_depth, float* out_alpha, float3* out_normal,
	float* out_median_depth, float* out_distortion, cudaStream_t stream)
{
    render<32, 16>(num_rendered, width, height, points_xy, rgb, other_features,
		conic_opacity, normal_opacity, Tu, Tv, Tw,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, out_other_features,
		out_depth, out_alpha, out_normal, out_median_depth, out_distortion, stream);
}

void render_32x32(int num_rendered,
	int width, int height,
	float2* points_xy, float3* rgb, float2* other_features,
	float4* conic_opacity, float4* normal_opacity,
	float3* Tu, float3* Tv, float3* Tw,
	uint64_t* gaussian_keys_sorted, uint32_t* gaussian_values_sorted,
	int2* ranges, float3 bg_color, float3* out_color, float2* out_other_features,
	float* out_depth, float* out_alpha, float3* out_normal,
	float* out_median_depth, float* out_distortion, cudaStream_t stream)
{
    render<32, 32>(num_rendered, width, height, points_xy, rgb, other_features,
		conic_opacity, normal_opacity, Tu, Tv, Tw,
	    gaussian_keys_sorted, gaussian_values_sorted, ranges, bg_color, out_color, out_other_features,
		out_depth, out_alpha, out_normal, out_median_depth, out_distortion, stream);
}

} // namespace flashgs