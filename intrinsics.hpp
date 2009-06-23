/*
 *	Copyright (c) 2009 Joel de Vahl
 *	              2009 Markus Ålind
 *	see LICENSE file for more info
 */

#ifndef INC_SIMD_INTRINSICS_HPP
#define INC_SIMD_INTRINSICS_HPP

#include <cmath>

#ifndef F32_INF
#       define F32_INF  static_cast<f32>(HUGE_VAL)
#endif

#ifndef F32_NINF
#       define F32_NINF static_cast<f32>(-HUGE_VAL)
#endif

#ifndef ALIGN
#	ifdefined(COMPILER_GCC)
#       	define ALIGN(_Align) __attribute__ ((aligned (_Align)))
#	elif defined(COMPILER_MSVC)
#       	define ALIGN(_Align) __declspec(align(_Align))
#	else
#       	error ALIGN not defined for this compiler
#	endif
#endif

#ifndef FORCE_INLINE
#	if defined(COMPILER_GCC)
#       	define FORCE_INLINE inline __attribute__((always_inline))
#	elif defined(COMPILER_MSVC)
#       	define FORCE_INLINE inline
#	else
#       	error FORCE_INLINE not defined for this compiler
#	endif
#endif

#ifndef RESTRICT
#	if defined(COMPILER_GCC)
#       	define RESTRICT __restrict__
#	elif defined(COMPILER_MSVC)
#       	define RESTRICT __restrict
#	else
#       	error RESTICT not defined for this compiler
#	endif
#endif

//#define SIMD_NONE
//#define SIMD_ALTIVEC
//#define SIMD_SSE 1
//#define SIMD_SSE 2
//#define SIMD_SSE 3
//#define SIMD_SSE 4

#if defined(SIMD_NONE)
	class v128
	{
	public:
		union
		{
			uint32_t k[4];
			float x, y, z, w;
		};

		v128() {}
		v128(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
		v128(float f) : x(f), y(f), z(f), w(f) {}
		v128(const v128 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		const v128 &operator=(const v128 &v) { x = v.x; y = v.y; z = v.z; w = v.w; }
	};

	class v128_u32
	{
	public:
		uint32_t x, y, z, w;

		v128() {}
		v128(uint32_t _x, uint32_t _y, uint32_t _z, uint32_t _w) : x(_x), y(_y), z(_z), w(_w) {}
		v128(uint32_t f) : x(f), y(f), z(f), w(f) {}
		v128(const v128 &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}
		const v128 &operator=(const v128 &v) { x = v.x; y = v.y; z = v.z; w = v.w; }
	};

#elif defined(SIMD_ALTIVEC)
#	include <altivec.h>
#	undef bool
#	undef pixel
	typedef vector float v128;



#elif defined(SIMD_SPU)
	typedef vector float v128;



#elif defined(SIMD_SSE)
#	include <xmmintrin.h>

#	if SIMD_SSE >= 2
#		include <emmintrin.h>
#	endif

#	if SIMD_SSE >= 3
#		include <pmmintrin.h>
#	endif

#	if SIMD_SSE >= 4
#		include <smmintrin.h>
#	endif

	typedef __m128 v128;

#else
	#error SIMD math not implemented for this platform
#endif

/*-----------------------------------------------------------------*\
 | ARITHMETIC OPERATION
\*-----------------------------------------------------------------*/

// These functions has variants for different types stored in v128
FORCE_INLINE v128 vAdd		(v128 v1, v128 v2);
FORCE_INLINE v128 vAdd_u32	(v128 v1, v128 v2);
FORCE_INLINE v128 vSub		(v128 v1, v128 v2);
FORCE_INLINE v128 vSub_u32	(v128 v1, v128 v2);
FORCE_INLINE v128 vMul		(v128 v1, v128 v2);
FORCE_INLINE v128 vMul_u32	(v128 v1, v128 v2);
FORCE_INLINE v128 vDiv		(v128 v1, v128 v2);
FORCE_INLINE v128 vDiv_u32	(v128 v1, v128 v2);

// These functions are defined for v128 as 4x float only
FORCE_INLINE v128 vMAdd		(v128 v1, v128 v2, v128 v3);
FORCE_INLINE v128 vDP3		(v128 v1, v128 v2);
FORCE_INLINE v128 vDP4		(v128 v1, v128 v2);
FORCE_INLINE v128 vSqrt		(v128 v);
FORCE_INLINE v128 vRce		(v128 v);
FORCE_INLINE v128 vRc			(v128 v);
FORCE_INLINE v128 vRsqe		(v128 v);
FORCE_INLINE v128 vRsq		(v128 v);

/*-----------------------------------------------------------------*\
 | COMPARSION OPERATIONS
\*-----------------------------------------------------------------*/

FORCE_INLINE v128 vCmpGE	(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpGE_u32(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpG		(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpG_u32	(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpLE	(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpLE_u32(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpL		(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpL_u32	(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpEQ	(v128 v1, v128 v2);
FORCE_INLINE v128 vCmpEQ_u32(v128 v1, v128 v2);

/*-----------------------------------------------------------------*\
 | BIT AND BYTE TWIDDLING
\*-----------------------------------------------------------------*/

// Agnostic to the type stored in v128
FORCE_INLINE v128 vRotR		(v128 v);
FORCE_INLINE v128 vRotR2	(v128 v);
FORCE_INLINE v128 vRotL		(v128 v);
FORCE_INLINE v128 vRotL2	(v128 v);
FORCE_INLINE v128 vSplat	(v128 v, uint8_t i);
FORCE_INLINE v128 vMuxLow	(v128 l, v128 h);
FORCE_INLINE v128 vMuxHigh	(v128 l, v128 h);
FORCE_INLINE v128 vMergeLow	(v128 l, v128 h);
FORCE_INLINE v128 vMergeHigh(v128 l, v128 h);
FORCE_INLINE v128 vOr		(v128 v1, v128 v2);
FORCE_INLINE v128 vAnd		(v128 v1, v128 v2);
FORCE_INLINE v128 vNot		(v128 v);
FORCE_INLINE v128 vShiftLeftByte(v128 v, uint32_t Bytes);

FORCE_INLINE v128 vSelMsk	(v128 m, v128 v1, v128 v2);
FORCE_INLINE v128 vMskW	(v128 _v, v128 _w);

// Implemented as macros
//FORCE_INLINE v128 vShuffle1(v128 v1, uint8_t _i0, uint8_t _i1, uint8_t _i2, uint8_t _i3);

/*-----------------------------------------------------------------*\
 | CONVERT, EXTRACT, LOAD AND STORE
\*-----------------------------------------------------------------*/

FORCE_INLINE float vExtract		(v128 v, uint8_t i);
FORCE_INLINE uint32_t vExtract_u32	(v128 v, uint8_t i);

FORCE_INLINE v128 vConv_s32float	(v128 v);
FORCE_INLINE v128 vConv_floats32	(v128 v);
FORCE_INLINE v128 vConv_floats32	(v128 v, v128 scale);

FORCE_INLINE v128 vInf		();
FORCE_INLINE v128 vNInf		();
FORCE_INLINE v128 vZero		();
FORCE_INLINE v128 vScalar	(float f);
FORCE_INLINE v128 vScalar_u32(uint32_t u);
FORCE_INLINE v128 vLoad		(float x, float y, float z, float w);
FORCE_INLINE v128 vLoad_u32	(uint32_t x, uint32_t y, uint32_t z, uint32_t w);
FORCE_INLINE v128 vLoad		(float *p);
FORCE_INLINE v128 vLoad_u32	(uint32_t *p);
FORCE_INLINE void vStore	(float *p, v128 v);
FORCE_INLINE void vStore_u32(uint32_t *p, v128 v);

/*-----------------------------------------------------------------*\
 | THE IMPLEMENTATION
\*-----------------------------------------------------------------*/


// macros

#if defined(SIMD_ALTIVEC)

#elif defined(SIMD_SPU)
#define D_SHUFFLE_MSK(_i0, _i1, _i2, _i3) ((v128)((vector unsigned char){ _i0*4, _i0*4+1, _i0*4+2, _i0*4+3,\
	_i1*4, _i1*4+1, _i1*4+2, _i1*4+3,\
	_i2*4+16, _i2*4+1+16, _i2*4+2+16, _i2*4+3+16,\
	_i3*4+16, _i3*4+1+16, _i3*4+2+16, _i3*4+3+16}))
#elif defined(SIMD_SSE)
#define D_SHUFFLE_MSK(_i0, _i1, _i2, _i3) ((v128)_MM_SHUFFLE(3-_i0, 3-_i1, 3-_i2, 3-_i3))
#else
	
#endif


#if defined(SIMD_ALTIVEC)
#define vShuffle2(v1, v2, _i0, _i1, _i2, _i3) \
		vec_perm(v1, v2, (vector unsigned char){ \
					_i0*4, _i0*4+1, _i0*4+2, _i0*4+3,\
					_i1*4, _i1*4+1, _i1*4+2, _i1*4+3,\
					_i2*4, _i2*4+1, _i2*4+2, _i2*4+3,\
					_i3*4, _i3*4+1, _i3*4+2, _i3*4+3,\
					})
#elif defined(SIMD_SPU)
	#define vShuffle2(v1, v2, _i0, _i1, _i2, _i3) \
		spu_shuffle(v1, v2, (vector unsigned char){ \
					_i0*4, _i0*4+1, _i0*4+2, _i0*4+3,\
					_i1*4, _i1*4+1, _i1*4+2, _i1*4+3,\
					_i2*4, _i2*4+1, _i2*4+2, _i2*4+3,\
					_i3*4, _i3*4+1, _i3*4+2, _i3*4+3,\
					})
#elif defined(SIMD_SSE)
#else
#endif

#if defined(SIMD_ALTIVEC)
	#define vMask(x, y, z, w) vLoad_u32(x * 0xffffffff, y * 0xffffffff, z * 0xffffffff, w * 0xffffffff)
#elif defined(SIMD_SPU)
	//#define vMask(x, y, z, w) vLoad_u32(x * 0xffffffff, y * 0xffffffff, z * 0xffffffff, w * 0xffffffff)
	#define vMask(x, y, z, w) ((v128) spu_maskb((u16)((~(1-x)&0xf000) | (~(1-y)&0xf00) | (~(1-z)&0xf0) | (~(1-w)&0xf))))
	//#define vMask(x, y, z, w) ((v128) spu_maskw((uint8_t)((x?8:0) | (y?4:0) | (z?2:0) | (w?1:0))))
#elif defined(SIMD_SSE)
	#define vMask(x, y, z, w) vLoad_u32(x * 0xffffffff, y * 0xffffffff, z * 0xffffffff, w * 0xffffffff)
#else
	#define vMask(x, y, z, w) vLoad_u32(x * 0xffffffff, y * 0xffffffff, z * 0xffffffff, w * 0xffffffff)
#endif

#if defined(SIMD_ALTIVEC) || defined(SIMD_SPU)
	#define vShuffle1(v, _i0, _i1, _i2, _i3) \
		vShuffle2(v, v, _i0, _i1, _i2, _i3)
#elif defined(SIMD_SSE)
	#define vShuffle1(v, _i0, _i1, _i2, _i3) \
		_mm_shuffle_ps(v, v, _MM_SHUFFLE(_i3, _i2, _i1, _i0))
#else
#endif

// Helper print functions

#define vDebugPrint(V) vDebugPrint_helper(#V, V)
#define vDebugPrint_u32(V) vDebugPrint_u32_helper(#V, V)
#define vDebugPrint_u32x(V) vDebugPrint_u32x_helper(#V, V)

#include <cstdio>

FORCE_INLINE void vDebugPrint_helper(const char* name, v128 v)
{	
	printf("%s = (%f %f %f %f)\n", name, vExtract(v, 0), vExtract(v, 1), vExtract(v, 2), vExtract(v, 3));
}

FORCE_INLINE void vDebugPrint_u32_helper(const char* name, v128 v)
{
	printf("%s = (%u %u %u %u)\n", name, vExtract_u32(v, 0), vExtract_u32(v, 1), vExtract_u32(v, 2), vExtract_u32(v, 3));
}
FORCE_INLINE void vDebugPrint_u32x_helper(const char* name, v128 v)
{
	printf("%s = (0x%x 0x%x 0x%x 0x%x)\n", name, vExtract_u32(v, 0), vExtract_u32(v, 1), vExtract_u32(v, 2), vExtract_u32(v, 3));
}

/*
	Function: vAdd
		Element wise addition of two vectors.

	Parameters:
		v1 - first vector
		v2 - second vector

	Returns:
		The two vectors added together.
*/
FORCE_INLINE v128 vAdd(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_add(v1, v2);
#elif defined(SIMD_SPU)
	return spu_add(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_add_ps(v1, v2);
#else
	return v128(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
#endif
}

FORCE_INLINE v128 vAdd_u32(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (v128)vec_add((vector unsigned int)v1, (vector unsigned int)v2);
#elif defined(SIMD_SPU)
	return (v128)spu_add((vector unsigned int)v1, (vector unsigned int)v2);
#elif defined(SIMD_SSE)
	return  _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(v1), _mm_castps_si128(v2)));
#else
	return v128(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z, v1.w+v2.w);
#endif
}

/*
	Function: vSub
		Element wise subtraction of two vectors.
*/
FORCE_INLINE v128 vSub(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_sub(v1, v2);
#elif defined(SIMD_SPU)
	return spu_sub(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_sub_ps(v1, v2);
#else
	return v128(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z, v1.w-v2.w);
#endif
}

/*
	Function: vMul
		Element wise multiplication of two vectors.
*/
FORCE_INLINE v128 vMul(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_madd(v1, v2, vZero());
#elif defined(SIMD_SPU)
	return spu_mul(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_mul_ps(v1, v2);
#else
	return v128(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w);
#endif
}

/*
	Function: vDiv
		Element wise division of two vectors.

	Parameters:
		v1 - first vector
		v2 - second vector

	Returns:
		The first vector divided by the second.
*/
FORCE_INLINE v128 vDiv(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_madd(v1, vRc(v2), vZero());
#elif defined(SIMD_SPU)
	return spu_mul(v1, vRc(v2));
#elif defined(SIMD_SSE)
	return _mm_div_ps(v1, v2);
#else
	return v128(v1.x/v2.x, v1.y/v2.y, v1.z/v2.z, v1.w/v2.w);
#endif
}

/*
	Function: vMadd
		Multiplies two vectors and adds a third. Operates element wise.

	Parameters:
		v1 - first vector
		v2 - second vector
		v3 - third vector (add)

	Returns:
		(v1 * v2) + v3
*/
FORCE_INLINE v128 vMAdd(v128 v1, v128 v2, v128 v3)
{
#if defined(SIMD_ALTIVEC)
	return vec_madd(v1, v2, v3);
#elif defined(SIMD_SPU)
	return spu_madd(v1, v2, v3);
#elif defined(SIMD_SSE)
	return _mm_add_ps(_mm_mul_ps(v1, v2), v3);
#else
	return v128(v1.x*v2.x+v3.x, v1.y*v2.y+v3.y, v1.z*v2.z+v3.z, v1.w*v2.w+v3.w);
#endif
}

/*
	Function: vDP3
		3 element dot product (scalar product) between two vectors.

	Parameters:
		v1 - first vector
		v2 - second vector

	Returns:
		Vector containing the dot product in all fields.
*/
FORCE_INLINE v128 vDP3(v128 v1, v128 v2)
{
//	TODO: Clean this function up
#if defined(SIMD_ALTIVEC)
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	return res;
#elif defined(SIMD_SPU)
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	return res;
#elif defined(SIMD_SSE)
#	if SIMD_SSE >= 4
	return _mm_dp_ps(v1, v2, 0x77);
#	else
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	return res;
#	endif
#else
	return v128(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
#endif
}

/*
	Function: vDP4
		4 element dot product (scalar product) between two vectors.

	Parameters:
		v1 - first vector
		v2 - second vector

	Returns:
		Vector containing the dot product in all fields.
*/
FORCE_INLINE v128 vDP4(v128 v1, v128 v2)
{
//	TODO: Clean this function up
#if defined(SIMD_ALTIVEC)
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	res = vAdd(res, vSplat(d, 3));
	return res;
#elif defined(SIMD_SPU)
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	res = vAdd(res, vSplat(d, 3));
	return res;
#elif defined(SIMD_SSE)
#	if SIMD_SSE >= 4
	return _mm_dp_ps(v1, v2, 0xff);
#	else
	const v128 d = vMul(v1, v2);
	v128 res = vSplat(d, 0);
	res = vAdd(res, vSplat(d, 1));
	res = vAdd(res, vSplat(d, 2));
	res = vAdd(res, vSplat(d, 3));
	return res;
#	endif
#else
	return v128(v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w);
#endif
}


/*
	Function: vSqrt
		Element wise square root of vector.
*/
FORCE_INLINE v128 vSqrt(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_madd(v, vRsq(v), vZero());
#elif defined(SIMD_SPU)
	return spu_madd(v, vRsq(v), vZero());
#elif defined(SIMD_SSE)
	return _mm_sqrt_ps(v);
#else
	return v128(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w));
#endif
}

/*
	Function: vRce
		Element wise reciprocal (1/x) estimate.
*/
FORCE_INLINE v128 vRce(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_re(v);
#elif defined(SIMD_SPU)
	return spu_re(v);
#elif defined(SIMD_SSE)
	return _mm_rcp_ps(v);
#else
	return v128(1.0f/v.x, 1.0f/v.y, 1.0f/v.z, 1.0f/v.w);
#endif
}

/*
	Function: vRc
		Element wise reciprocal (1/x).
		Calulated using the reciprocal estimate and performing a one step Newton-Raphson refinement.
*/
FORCE_INLINE v128 vRc(v128 v)
{
#if defined(SIMD_ALTIVEC)
	v128 e = vec_re(v); // initial estimate
	v128 r = vec_madd(vec_nmsub(e, v, vScalar(1.0f)), e, e); // one step of newton-raphson
	vector __bool int not_nan = vec_cmpeq(r, r); // did we get a NaN in the previous step?
	return vec_sel(e, r, not_nan); // use the estimate (-Inf or Inf) instead of NaN
#elif defined(SIMD_SPU)
	v128 e = spu_re(v); // initial estimate
	v128 r = spu_madd(spu_nmsub(e, v, vScalar(1.0f)), e, e); // one step of newton-raphson
	vector unsigned not_nan = spu_cmpeq(r, r); // did we get a NaN in the previous step?
	return spu_sel(e, r, not_nan); // use the estimate (-Inf or Inf) instead of NaN
#elif defined(SIMD_SSE)
//	TODO: Is this fast or slow?
	return vDiv(vScalar(1.0f), v);
#else
	return v128(1.0f/v.x, 1.0f/v.y, 1.0f/v.z, 1.0f/v.w);
#endif
}

/*
	Function: vRsqe
		Element wise reciprocal square root (1/sqrt(x)) estimate.
*/
FORCE_INLINE v128 vRsqe(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_rsqrte(v);
#elif defined(SIMD_SPU)
	return spu_rsqrte(v);
#elif defined(SIMD_SSE)
	return _mm_rsqrt_ps(v);
#else
	return v128(1.0f/sqrt(v.x), 1.0f/sqrt(v.y), 1.0f/sqrt(v.z), 1.0f/sqrt(v.w));
#endif
}

/*
	Function: vRsq
		Element wise reciprocal square root (1/sqrt(x)).
		Calulated using the reciprocal square root estimate and performing a one step Newton-Raphson refinement.
*/
FORCE_INLINE v128 vRsq(v128 v)
{
#if defined(SIMD_ALTIVEC)
	v128 zero = vZero();
	v128 half = vScalar(0.5f);
	v128 one =  vScalar(1.0f);
	v128 e = vec_rsqrte(v); // initial estimate
	v128 es = vec_madd(e, e, zero);
	v128 eh = vec_madd(e, half, zero);
	return vec_madd(vec_nmsub(v, es, one), eh, e); // one step newton-raphson
#elif defined(SIMD_SPU)
	v128 zero = vZero();
	v128 half = vScalar(0.5f);
	v128 one =  vScalar(1.0f);
	v128 e = spu_rsqrte(v); // initial estimate
	v128 es = spu_madd(e, e, zero);
	v128 eh = spu_madd(e, half, zero);
	return spu_madd(spu_nmsub(v, es, one), eh, e); // one step newton-raphson
#elif defined(SIMD_SSE)
//	TODO: Truly horrible
	return vDiv(vScalar(1.0f), _mm_sqrt_ps(v));
#else
	return v128(1.0f/sqrt(v.x), 1.0f/sqrt(v.y), 1.0f/sqrt(v.z), 1.0f/sqrt(v.w));
#endif
}

#define MIN(x,y) x <= y ? x : y
#define MAX(x,y) x >= y ? x : y

/*
	Function: vMin
		Element wise min of two vectors
*/
FORCE_INLINE v128 vMin(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_min(v1, v2);
#elif defined(SIMD_SPU)
	return spu_sel(v1, v2, spu_cmpgt(v1, v2));
#elif defined(SIMD_SSE)
	return _mm_min_ps(v1, v2);
#else
	return v128(
			MIN(v1.x, v2.x),
			MIN(v1.y, v2.y),
			MIN(v1.z, v2.z),
			MIN(v2.w, v2.w)
			);
#endif
}

/*
	Function: vMax
		Element wise max of two vectors
*/
FORCE_INLINE v128 vMax(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_max(v1, v2);
#elif defined(SIMD_SPU)
	return spu_sel(v2, v1, spu_cmpgt(v1, v2));
#elif defined(SIMD_SSE)
	return _mm_max_ps(v1, v2);
#else
	return v128(
			MAX(v1.x, v2.x),
			MAX(v1.y, v2.y),
			MAX(v1.z, v2.z),
			MAX(v2.w, v2.w)
			);
#endif
}

FORCE_INLINE v128 vCmpGE(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cmpge(v1, v2);
#elif defined(SIMD_SPU)
	return (vector float)spu_or(spu_cmpgt(v1, v2), spu_cmpeq(v1, v2));
#elif defined(SIMD_SSE)
	return _mm_cmpge_ps(v1, v2);
#else
	return v128(
			v1.x >= v2.x ? 1 : 0,
			v1.y >= v2.z ? 1 : 0,
			v1.z >= v2.y ? 1 : 0,
			v1.w >= v2.w ? 1 : 0
			);
#endif
}

FORCE_INLINE v128 vCmpG(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cmpgt(v1, v2);
#elif defined(SIMD_SPU)
	return (vector float)spu_cmpgt(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_cmpgt_ps(v1, v2);
#else
	return v128(
			v1.x > v2.x ? 1 : 0,
			v1.y > v2.z ? 1 : 0,
			v1.z > v2.y ? 1 : 0,
			v1.w > v2.w ? 1 : 0
			);
#endif
}

FORCE_INLINE v128 vCmpLE(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cmple(v1, v2);
#elif defined(SIMD_SPU)
	return (vector float)spu_xor(spu_cmpgt(v1, v2), (vector unsigned int)spu_splats((unsigned char)0xff));
#elif defined(SIMD_SSE)
	return _mm_cmple_ps(v1, v2);
#else
	return v128(
			v1.x <= v2.x ? 1 : 0,
			v1.y <= v2.z ? 1 : 0,
			v1.z <= v2.y ? 1 : 0,
			v1.w <= v2.w ? 1 : 0
			);
#endif
}

FORCE_INLINE v128 vCmpL(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cmplt(v1, v2);
#elif defined(SIMD_SPU)
	return (vector float)spu_nor(spu_cmpgt(v1, v2), spu_cmpeq(v1, v2));
#elif defined(SIMD_SSE)
	return _mm_cmplt_ps(v1, v2);
#else
	return v128(
			v1.x < v2.x ? 1 : 0,
			v1.y < v2.z ? 1 : 0,
			v1.z < v2.y ? 1 : 0,
			v1.w < v2.w ? 1 : 0
			);
#endif
}

FORCE_INLINE v128 vCmpEQ(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cmpeq(v1, v2);
#elif defined(SIMD_SPU)
	return (vector float)spu_cmpeq(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_cmpeq_ps(v1, v2);
#else
	return v128(
			v1.x == v2.x ? 1 : 0,
			v1.y == v2.z ? 1 : 0,
			v1.z == v2.y ? 1 : 0,
			v1.w == v2.w ? 1 : 0
			);
#endif
}

FORCE_INLINE v128 vRotR(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(v, v, (vector unsigned char){
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7,
				 8,  9, 10, 11
				 });
#elif defined(SIMD_SPU)
	return spu_shuffle(v, v, (vector unsigned char){
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7,
				 8,  9, 10, 11
				 });
#elif defined(SIMD_SSE)
	return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 0, 1, 2));
#else
	return v128(v.w, v.x, v.y, v.z);
#endif
}

FORCE_INLINE v128 vRotR2(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(v, v, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7
				 });
#elif defined(SIMD_SPU)
	return spu_shuffle(v, v, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7
				 });
#elif defined(SIMD_SSE)
	return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
#else
	return v128(v.z, v.w, v.x, v.y);
#endif
}

FORCE_INLINE v128 vRotL(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(v, v, (vector unsigned char){
				 4,  5,  6,  7,
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3
				 });
#elif defined(SIMD_SPU)
	return spu_shuffle(v, v, (vector unsigned char){
				 4,  5,  6,  7,
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3
				 });
#elif defined(SIMD_SSE)
	return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 2, 3, 0));
#else
	return v128(v.y, v.z, v.w, v.x);
#endif
}

FORCE_INLINE v128 vRotL2(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(v, v, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7
				 });
#elif defined(SIMD_SPU)
	return spu_shuffle(v, v, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				 0,  1,  2,  3,
				 4,  5,  6,  7
				 });
#elif defined(SIMD_SSE)
	return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
#else
	return v128(v.z, v.w, v.x, v.y);
#endif
}

FORCE_INLINE v128 vSplat(v128 v, uint8_t i)
{
#if defined(SIMD_ALTIVEC)
	switch(i)
	{
	case 0:
		return vec_splat(v, 0);
	case 1:
		return vec_splat(v, 1);
	case 2:
		return vec_splat(v, 2);
	}

	return vec_splat(v, 3);
#elif defined(SIMD_SPU)
	//return vShuffle1(v, i,i,i,i);
	return spu_splats(spu_extract(v, i));
/*	switch(i)
	{
	case 0:
		return spu_splats(spu_extract(v, 0));
	case 1:
		return spu_splats(spu_extract(v, 1));
	case 2:
		return spu_splats(spu_extract(v, 2));
	}

	return spu_splats(spu_extract(v, 3));*/
#elif defined(SIMD_SSE)
	switch(i)
	{
	case 0:
		return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	case 1:
		return _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	case 2:
		return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	}

	return _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
#else
	switch(i)
	{
	case 0:
		return v128(v.x);
	case 1:
		return v128(v.y);
	case 2:
		return v128(v.z);
	}
	
	return v128(v.w);
#endif
}

// vMergeLow(vLoad(0,1,2,3), vLoad(4,5,6,7)) = (0.000000 4.000000 1.000000 5.000000)
FORCE_INLINE v128 vMergeLow(v128 l, v128 h)
{
#if defined(SIMD_ALTIVEC)
	return vec_mergel(l, h);
#elif defined(SIMD_SPU)
	return spu_shuffle(l, h, (vector unsigned char){
				 8,  9, 10, 11,
				24, 25, 26, 27,
				12, 13, 14, 15,
				28, 29, 30, 31
				});
#elif defined(SIMD_SSE)
	return _mm_unpackhi_ps(l, h);
#else
	return v128(l.z, h.z, l.w, h.w);
#endif
}

// vMergeHigh(vLoad(0,1,2,3), vLoad(4,5,6,7)) = (2.000000 6.000000 3.000000 7.000000)
FORCE_INLINE v128 vMergeHigh(v128 l, v128 h)
{
#if defined(SIMD_ALTIVEC)
	return vec_mergeh(l, h);
#elif defined(SIMD_SPU)
	return spu_shuffle(l, h, (vector unsigned char){
				 0,  1,  2,  3,
				16, 17, 18, 18,
				 4,  5,  6,  7,
				20, 21, 22, 23
				});	
#elif defined(SIMD_SSE)
	return _mm_unpacklo_ps(l, h);
#else
	return v128(l.x, h.x, l.y, h.y);
#endif
}

FORCE_INLINE v128 vMuxLow(v128 l, v128 h)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(l, h, (vector unsigned char){
				 0,  1,  2,  3,
				 4,  5,  6,  7,
				16, 17, 18, 18,
				20, 21, 22, 23
				});
#elif defined(SIMD_SPU)
	return spu_shuffle(l, h, (vector unsigned char){
				 0,  1,  2,  3,
				 4,  5,  6,  7,
				16, 17, 18, 18,
				20, 21, 22, 23
				});
#elif defined(SIMD_SSE)
	return _mm_movelh_ps(l, h);
#else
	return v128(l.x, l.y, h.x, h.y);
#endif
}

FORCE_INLINE v128 vMuxHigh(v128 l, v128 h)
{
#if defined(SIMD_ALTIVEC)
	return vec_perm(l, h, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				24, 25, 26, 27,
				28, 29, 30, 31
				});
#elif defined(SIMD_SPU)
	return spu_shuffle(l, h, (vector unsigned char){
				 8,  9, 10, 11,
				12, 13, 14, 15,
				24, 25, 26, 27,
				28, 29, 30, 31
				});
#elif defined(SIMD_SSE)
	return _mm_movehl_ps(l, h);
#else
	return v128(l.z, l.w, h.z, h.w);
#endif
}

FORCE_INLINE v128 vSelMsk(v128 m, v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_sel(v2, v1, (vector unsigned int)m);
#elif defined(SIMD_SPU)
	return spu_sel(v2, v1, (vector unsigned int)m);
#elif defined(SIMD_SSE)
	return vOr(vAnd(m, v1), _mm_andnot_ps(m, v2));
#else
#endif
}


FORCE_INLINE v128 vMskW(v128 _v, v128 _w)
{
	return vSelMsk(vMask(1,1,1,0), _v, _w);
}

FORCE_INLINE v128 vOr(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_or(v1, v2);
#elif defined(SIMD_SPU)
	return spu_or(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(v1), _mm_castps_si128(v2)));
#else
	return v128(v1.x | v2.x, v1.x | v2.x, v1.z | v2.z, v1.w | v2.w);
#endif
}

FORCE_INLINE v128 vAnd(v128 v1, v128 v2)
{
#if defined(SIMD_ALTIVEC)
	return vec_and(v1, v2);
#elif defined(SIMD_SPU)
	return spu_and(v1, v2);
#elif defined(SIMD_SSE)
	return _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(v1), _mm_castps_si128(v2)));
#else
	return v128(v1.x & v2.x, v1.x & v2.x, v1.z & v2.z, v1.w & v2.w);
#endif
}

FORCE_INLINE v128 vNot(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_xor(v, vScalar_u32(0xffffffff));
#elif defined(SIMD_SPU)
	return spu_nand(v, v);
#elif defined(SIMD_SSE)
	return _mm_xor_ps(v, vScalar_u32(0xffffffff));
#else
	return v128(~v.x, ~v.x, ~v.z, ~v.w);
#endif
}

FORCE_INLINE v128 vShiftLeftByte(v128 v, uint32_t Bytes)
{
#if defined(SIMD_ALTIVEC)
	switch(Bytes)
	{
		case 0:	return vec_sld(v,v, 0);
		case 1:	return vec_sld(v,v, 1);
		case 2:	return vec_sld(v,v, 2);
		case 3:	return vec_sld(v,v, 3);
	}
#elif defined(SIMD_SPU)
#elif defined(SIMD_SSE)
	switch(Bytes)
	{
	case 0: return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 0));
	case 1: return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 1));
	case 2: return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 2));
	}
	return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 3));
#else
	return v128(v.x << (Bytes*8), v.x << (Bytes*8), v.z << (Bytes*8), v.w << (Bytes*8));
#endif
}

FORCE_INLINE uint32_t vExtract_u32(v128 v, uint8_t i)
{
#if defined(SIMD_ALTIVEC)
	uint32_t lU[4];
	vStore_u32(&lU[0], v);
	return lU[i];
	//return vec_extract(v, i);
#elif defined(SIMD_SPU)
	return spu_extract((vector unsigned int)v, i);
#elif defined(SIMD_SSE)
	switch(i)
	{
		case 0: return _mm_cvtsi128_si32(_mm_shuffle_epi32((__m128i&)v, _MM_SHUFFLE(0,0,0,0)));
	 	case 1: return _mm_cvtsi128_si32(_mm_shuffle_epi32((__m128i&)v, _MM_SHUFFLE(0,0,0,1)));
		case 2: return _mm_cvtsi128_si32(_mm_shuffle_epi32((__m128i&)v, _MM_SHUFFLE(0,0,0,2)));
		case 3: return _mm_cvtsi128_si32(_mm_shuffle_epi32((__m128i&)v, _MM_SHUFFLE(0,0,0,3)));
	}

#else
#endif
}


FORCE_INLINE float vExtract(v128 v, uint8_t i)
{
#if defined(SIMD_ALTIVEC)
	float lF[4];
	vStore(&lF[0], v);
	return lF[i];
//	return vec_extract(v, i);
#elif defined(SIMD_SPU)
	return spu_extract(v, i);
#elif defined(SIMD_SSE)
	switch(i)
	{
		case 0:	return _mm_cvtss_float(v);
		case 1:	return _mm_cvtss_float(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,1)));
		case 2:	return _mm_cvtss_float(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,2)));
		case 3:	return _mm_cvtss_float(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,3)));
	}

#else
#endif
}

FORCE_INLINE v128 vConv_floats32(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_cts(v, 0);
#elif defined(SIMD_SPU)
	return (vector float)spu_convts(v, 0);
#elif defined(SIMD_SSE)
	return _mm_castsi128_ps(_mm_cvtps_epi32(v));
#else
#endif
}

FORCE_INLINE v128 vConv_floatu32(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_ctu(v, 0);
#elif defined(SIMD_SPU)
	return (vector float)spu_convtu(v, 0);
#elif defined(SIMD_SSE)
	return _mm_castsi128_ps(_mm_cvttps_epi32(vMax(v, vScalar(0.0f))));
#else
#endif
}

FORCE_INLINE v128 vConv_s32float(v128 v)
{
#if defined(SIMD_ALTIVEC)
	return vec_ctf((vector unsigned int)v, 0);
#elif defined(SIMD_SPU)
	return spu_convtf((vector unsigned int)v, 0);
#elif defined(SIMD_SSE)
	return _mm_cvtepi32_ps(_mm_castps_si128(v));
#else
#endif
}

/*
FORCE_INLINE v128 vConvfloats32(v128 v)
{
#if defined(SIMD_ALTIVEC)
#elif defined(SIMD_SPU)
#elif defined(SIMD_SSE)
#else
#endif
}


FORCE_INLINE v128 vConvfloats32(v128 v)
{
#if defined(SIMD_ALTIVEC)
#elif defined(SIMD_SPU)
#elif defined(SIMD_SSE)
#else
#endif
}

FORCE_INLINE v128 vConvfloats32(v128 v)
{
#if defined(SIMD_ALTIVEC)
#elif defined(SIMD_SPU)
#elif defined(SIMD_SSE)
#else
#endif
}
*/



FORCE_INLINE v128 vInf()
{
#if defined(SIMD_ALTIVEC)
	return vScalar(F32_INF);
#elif defined(SIMD_SPU)
	return vScalar(F32_INF);
#elif defined(SIMD_SSE)
	return vScalar(F32_INF);
#else
	return v128(F32_INF);
#endif
}

FORCE_INLINE v128 vNInf()
{
#if defined(SIMD_ALTIVEC)
	return vScalar(F32_NINF);
#elif defined(SIMD_SPU)
	return vScalar(F32_NINF);
#elif defined(SIMD_SSE)
	return vScalar(F32_NINF);
#else
	return v128(F32_NINF);
#endif
}

FORCE_INLINE v128 vZero()
{
#if defined(SIMD_ALTIVEC)
	return (v128){0.0f, 0.0f, 0.0f, 0.0f};
#elif defined(SIMD_SPU)
	return (v128){0.0f, 0.0f, 0.0f, 0.0f};
#elif defined(SIMD_SSE)
	return  _mm_setzero_ps();
#else
	return v128(0.0f);
#endif
}

FORCE_INLINE v128 vScalar(float f)
{
#if defined(SIMD_ALTIVEC)
	return (v128){f, f, f, f};
#elif defined(SIMD_SPU)
	return spu_splats(f);
#elif defined(SIMD_SSE)
	return _mm_set1_ps(f);
#else
	return v128(f);
#endif
}

FORCE_INLINE v128 vScalar_u32(uint32_t u)
{
#if defined(SIMD_ALTIVEC)
	return (v128)((vector unsigned int){u, u, u, u});
#elif defined(SIMD_SPU)
	return (v128)spu_splats(u);
#elif defined(SIMD_SSE)
	const uint32_t ALIGN(16) p[4] = {u, u, u, u};
	return _mm_castsi128_ps(_mm_load_si128((__m128i*)p));
#else
	return v128(u);
#endif
}

FORCE_INLINE v128 vLoad(float x, float y, float z, float w)
{
#if defined(SIMD_ALTIVEC)
	return (v128){x, y, z, w};
#elif defined(SIMD_SPU)
	return (v128){x, y, z, w};
#elif defined(SIMD_SSE)
	const float ALIGN(16) p[4] = {x,y,z,w};
	return _mm_load_ps(p);
#else
	return v128(x, y, z, w);
#endif
}

FORCE_INLINE v128 vLoad_u32(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
#if defined(SIMD_ALTIVEC)
	return (v128)((vector unsigned int){x, y, z, w});
#elif defined(SIMD_SPU)
	return (v128)((vector unsigned int){x, y, z, w});
#elif defined(SIMD_SSE)
	const uint32_t ALIGN(16) p[4] = {x,y,z,w};
	return _mm_castsi128_ps(_mm_load_si128((__m128i*)p));
#else
	return v128(x, y, z, w);
#endif
}

FORCE_INLINE v128 vLoad(float *p)
{
#if defined(SIMD_ALTIVEC)
	return vec_ld(0, p);
#elif defined(SIMD_SPU)
	vector float r = *(vector float *) p;
	return r;
#elif defined(SIMD_SSE)
	return _mm_load_ps(p);
#else
	return v128(p[0], p[1], p[2], p[3]);
#endif
}

FORCE_INLINE v128 vLoad_u32(uint32_t *p)
{
#if defined(SIMD_ALTIVEC)
	return (vector float)vec_ld(0, p);
#elif defined(SIMD_SPU)
	vector unsigned int r = *(vector unsigned int *) p;
	return (vector float)r;
#elif defined(SIMD_SSE)
	return _mm_castsi128_ps(_mm_load_si128((__m128i*)p));
#else
	return v128(p[0], p[1], p[2], p[3]);
#endif
}

FORCE_INLINE void vStore(float *p, v128 v)
{
#if defined(SIMD_ALTIVEC)
	vec_st(v, 0, p);
#elif defined(SIMD_SPU)
	float * pv = (float *)&v;
	p[0] = pv[0];
	p[1] = pv[1];
	p[2] = pv[2];
	p[3] = pv[3];
#elif defined(SIMD_SSE)
	_mm_store_ps(p, v);
#else
	p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
#endif
}

FORCE_INLINE void vStore_u32(uint32_t *p, v128 v)
{
#if defined(SIMD_ALTIVEC)
	vec_st((vector unsigned int)v, 0, p);
#elif defined(SIMD_SPU)
	unsigned int * pv = (unsigned int *)&v;
	p[0] = pv[0];
	p[1] = pv[1];
	p[2] = pv[2];
	p[3] = pv[3];
#elif defined(SIMD_SSE)
	_mm_store_si128((__m128i*)p, (__m128i&)v);
#else
	p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
#endif
}



#endif
