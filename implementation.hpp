#ifndef INC_SIMD_IMPLEMENTATION_HPP
#define INC_SIMD_IMPLEMENTATION_HPP
#include "intrinsics.hpp"

/*-----------------------------------------------------------------*\
 | Complex bit and byte twiddling
\*-----------------------------------------------------------------*/

FORCE_INLINE v128 vCross3(v128 v1, v128 v2);

FORCE_INLINE void mTranspose(const v128 &_rvS0,const v128 &_rvS1, const v128 &_rvS2, const v128 &_rvS3, v128 &_rvD0, v128 &_rvD1, v128 &_rvD2, v128 &_rvD3);
FORCE_INLINE void mTranspose(v128 &_rv0, v128 &_rv1, v128 &_rv2, v128 &_rv3);
FORCE_INLINE void mTranspose(const v128 *_rvSrc, v128 &_rv0, v128 &_rv1, v128 &_rv2, v128 &_rv3);
//FORCE_INLINE void mTranspose(v128 *_rvSrc, v128 *_rvDst);

FORCE_INLINE v128 mMulV(const v128 &_rvMS0, const v128 &_rvMS1, const v128 &_rvMS2, const v128 &_rvMS3, const v128 &_rvVS);
FORCE_INLINE v128 mMulVT(const v128 &_rvMS0, const v128 &_rvMS1, const v128 &_rvMS2, const v128 &_rvMS3, const v128 &_rvVS);
FORCE_INLINE void mMulM(const v128 &_rvM1S0, const v128 &_rvM1S1, const v128 &_rvM1S2, const v128 &_rvM1S3,
						const v128 &_rvM2S0, const v128 &_rvM2S1, const v128 &_rvM2S2, const v128 &_rvM2S3,
						v128 &_rvMD0, v128 &_rvMD1, v128 &_rvMD2, v128 &_rvMD3);

void FORCE_INLINE mFromPosFwdUp_Slow(const v128& _rvPos, const v128& _rvFwd, const v128& _rvUp, 
	v128& _rM0Dst, v128& _rM1Dst, v128& _rM2Dst, v128& _rM3Dst);
void FORCE_INLINE mFromVectors_Slow(const v128& _rvTrans, const v128& _rvX, const v128& _rvY, const v128& _rvZ, 
	v128& _rM0Dst, v128& _rM1Dst, v128& _rM2Dst, v128& _rM3Dst);

/*-----------------------------------------------------------------*\
 | Math utility functions
\*-----------------------------------------------------------------*/

FORCE_INLINE v128 vAddH(v128 _v);
FORCE_INLINE v128 vAddH3(v128 _v);
FORCE_INLINE v128 vAddH4x4(v128 _v0, v128 _v1, v128 _v2, v128 _v3);
FORCE_INLINE v128 vAddH4x3(v128 _v0, v128 _v1, v128 _v2, v128 _v3);

FORCE_INLINE v128 vNormalize(v128 _v);
FORCE_INLINE void vNormalize4x3(v128 _v0, v128 _v1, v128 _v2, v128 _v3);
FORCE_INLINE void vNormalize4x4(v128 _v0, v128 _v1, v128 _v2, v128 _v3);
FORCE_INLINE v128 vNormalize3(v128 _v);

/*-----------------------------------------------------------------*\
 | THE IMPLEMENTATION
\*-----------------------------------------------------------------*/


/*
	Function: vCross3
		Cross product (vector product) of the first 3 elements of two vectors.

	Parameters:
		v1 - first vector
		v2 - second vector

	Returns:
		Cross product vector, w is carried from v1.
*/
FORCE_INLINE v128 vCross3(v128 v1, v128 v2)
{
	v128 v1yzxw = vShuffle1(v1, 1, 2, 0, 3);
	v128 v2zxyw = vShuffle1(v2, 2, 0, 1, 3);
	v128 vRes0 = vMul(v1yzxw, v2zxyw);
	v128 v1zxyw = vShuffle1(v1, 2, 0, 1, 3);
	v128 v2yzxw = vShuffle1(v2, 1, 2, 0, 3);
	v128 vRes1 = vMul(v1zxyw, v2yzxw);
	v128 vFin = vSub(vRes0, vRes1);
	return vMskW(vFin, v1);
}


FORCE_INLINE void mTranspose(const v128 &_rvS0,const v128 &_rvS1, const v128 &_rvS2, const v128 &_rvS3, v128 &_rvD0, v128 &_rvD1, v128 &_rvD2, v128 &_rvD3)
{
	const v128 v0 = _rvS0;
	const v128 v1 = _rvS1;
	const v128 v2 = _rvS2;
	const v128 v3 = _rvS3;
	
	const v128 vTmp0 = vMergeHigh(v0, v2);
	const v128 vTmp1 = vMergeLow(v0, v2);
	const v128 vTmp2 = vMergeHigh(v1, v3);
	const v128 vTmp3 = vMergeLow(v1, v3);

	const v128 vTrans3 = vMergeLow(vTmp1, vTmp3);
	const v128 vTrans2 = vMergeHigh(vTmp1, vTmp3);
	const v128 vTrans1 = vMergeLow(vTmp0, vTmp2);
	const v128 vTrans0 = vMergeHigh(vTmp0, vTmp2);

	_rvD0 = vTrans0;
	_rvD1 = vTrans1;
	_rvD2 = vTrans2;
	_rvD3 = vTrans3;
}

FORCE_INLINE void mTranspose(v128 &_rv0, v128 &_rv1, v128 &_rv2, v128 &_rv3)
{
	mTranspose(_rv0, _rv1, _rv2, _rv3, _rv0, _rv1, _rv2, _rv3);
}

FORCE_INLINE void mTranspose(const v128 *_rvSrc, v128 &_rv0, v128 &_rv1, v128 &_rv2, v128 &_rv3)
{	
	const v128 vS0 = _rvSrc[0];
	const v128 vS1 = _rvSrc[1];
	const v128 vS2 = _rvSrc[2];
	const v128 vS3 = _rvSrc[3];
	mTranspose(vS0, vS1, vS2, vS3, _rv0, _rv1, _rv2, _rv3);
}

/*
FORCE_INLINE void mTranspose	(v128 *_rvSrc, v128 *_rvDst)
{
#if defined(SIMD_ALTIVEC)
#elif defined(SIMD_SPU)
#elif defined(SIMD_SSE)
	const __m64* __restrict__ src = reinterpret_cast<const __m64* __restrict__>(_rvSrc);
	__m128 tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, src), &src[2]);
	__m128 row1 = _mm_loadh_pi(_mm_loadl_pi(row1, &src[4]), &src[6]);
	__m128 row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
	row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);
	tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, &src[1]), &src[3]);
	__m128 row3 = _mm_loadh_pi(_mm_loadl_pi(row3, &src[5]), &src[7]);
	__m128 row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
	row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);
	
#else
#endif

} */

FORCE_INLINE v128 mMulV(const v128 &_rvMS0, const v128 &_rvMS1, const v128 &_rvMS2, const v128 &_rvMS3, const v128 &_rvVS)
{
	v128 t0, t1, t2, t3;
	mTranspose(_rvMS0, _rvMS1, _rvMS2, _rvMS3, t0, t1, t2, t3);
	return mMulVT(t0, t1, t2, t3, _rvVS);
}

FORCE_INLINE v128 mMulVT(const v128 &_rvMS0, const v128 &_rvMS1, const v128 &_rvMS2, const v128 &_rvMS3, const v128 &_rvVS)
{
	
	v128 vVD = vMul(vSplat(_rvVS, 0), _rvMS0);
	vVD = vMAdd(vSplat(_rvVS, 1), _rvMS1, vVD);
	vVD = vMAdd(vSplat(_rvVS, 2), _rvMS2, vVD);
	vVD = vMAdd(vSplat(_rvVS, 3), _rvMS3, vVD);
	return vVD;
}

FORCE_INLINE void mMulM(const v128 &_rvM1S0, const v128 &_rvM1S1, const v128 &_rvM1S2, const v128 &_rvM1S3,
						const v128 &_rvM2S0, const v128 &_rvM2S1, const v128 &_rvM2S2, const v128 &_rvM2S3,
						v128 &_rvMD0, v128 &_rvMD1, v128 &_rvMD2, v128 &_rvMD3)
{
	_rvMD0 = vMAdd(vSplat(_rvM1S0, 0), _rvM2S0, vZero());
	_rvMD1 = vMAdd(vSplat(_rvM1S1, 0), _rvM2S0, vZero());
	_rvMD2 = vMAdd(vSplat(_rvM1S2, 0), _rvM2S0, vZero());
	_rvMD3 = vMAdd(vSplat(_rvM1S3, 0), _rvM2S0, vZero());

	_rvMD0 = vMAdd(vSplat(_rvM1S0, 1), _rvM2S1, _rvMD0);
	_rvMD1 = vMAdd(vSplat(_rvM1S1, 1), _rvM2S1, _rvMD1);
	_rvMD2 = vMAdd(vSplat(_rvM1S2, 1), _rvM2S1, _rvMD2);
	_rvMD3 = vMAdd(vSplat(_rvM1S3, 1), _rvM2S1, _rvMD3);

	_rvMD0 = vMAdd(vSplat(_rvM1S0, 2), _rvM2S2, _rvMD0);
	_rvMD1 = vMAdd(vSplat(_rvM1S1, 2), _rvM2S2, _rvMD1);
	_rvMD2 = vMAdd(vSplat(_rvM1S2, 2), _rvM2S2, _rvMD2);
	_rvMD3 = vMAdd(vSplat(_rvM1S3, 2), _rvM2S2, _rvMD3);

	_rvMD0 = vMAdd(vSplat(_rvM1S0, 3), _rvM2S3, _rvMD0);
	_rvMD1 = vMAdd(vSplat(_rvM1S1, 3), _rvM2S3, _rvMD1);
	_rvMD2 = vMAdd(vSplat(_rvM1S2, 3), _rvM2S3, _rvMD2);
	_rvMD3 = vMAdd(vSplat(_rvM1S3, 3), _rvM2S3, _rvMD3);
}


void FORCE_INLINE mFromVectors_Slow(const v128& _rvTrans, const v128& _rvX, const v128& _rvY, const v128& _rvZ, 
	v128& _rM0Dst, v128& _rM1Dst, v128& _rM2Dst, v128& _rM3Dst)
{
	f32* rfTrans = (f32*)&_rvTrans;
	f32* rfX = (f32*)&_rvX;
	f32* rfY = (f32*)&_rvY;
	f32* rfZ = (f32*)&_rvZ;
	
	/*matrix.M11 = x_axis.X;
	matrix.M12 = x_axis.Y;
	matrix.M13 = x_axis.Z;

	matrix.M21 = y_axis.X;
	matrix.M22 = y_axis.Y;
	matrix.M23 = y_axis.Z;

	matrix.M31 = z_axis.X;
	matrix.M32 = z_axis.Y;
	matrix.M33 = z_axis.Z;*/
	
	_rM0Dst = vLoad(rfX[0], rfY[0], rfZ[0], rfTrans[0]);
	_rM1Dst = vLoad(rfX[1], rfY[1], rfZ[1], rfTrans[1]);
	_rM2Dst = vLoad(rfX[2], rfY[2], rfZ[2], rfTrans[2]);
	_rM3Dst = vLoad(0.0f, 0.0f, 0.0f, 1.0f);
}

void FORCE_INLINE mFromPosFwdUp_Slow(const v128& _rvPos, const v128& _rvFwd, const v128& _rvUp, 
	v128& _rM0Dst, v128& _rM1Dst, v128& _rM2Dst, v128& _rM3Dst)
{	
//	vDebugPrint(_rvUp);
//	vDebugPrint(_rvFwd);
	v128 vX = vCross3(_rvUp, _rvFwd);
//	vDebugPrint(vX);
	mFromVectors_Slow(_rvPos, vX, _rvUp, _rvFwd, _rM0Dst, _rM1Dst, _rM2Dst, _rM3Dst);
}



/*-----------------------------------------------------------------*\
 | Math utility functions
\*-----------------------------------------------------------------*/

FORCE_INLINE v128 vAddH(v128 _v)
{
	v128 v0 = vSplat(_v, 0);
	v128 v1 = vSplat(_v, 1);
	v128 v2 = vSplat(_v, 2);
	v128 v3 = vSplat(_v, 3);
	return vAdd(vAdd(v0, v1), vAdd(v2, v3));
}

FORCE_INLINE v128 vAddH3(v128 _v)
{
	v128 v0 = vSplat(_v, 0);
	v128 v1 = vSplat(_v, 1);
	v128 v2 = vSplat(_v, 2);
	return vAdd(vAdd(v0, v1), v2);
}

FORCE_INLINE v128 vAddH4x4(v128 _v0, v128 _v1, v128 _v2, v128 _v3)
{
	mTranspose(_v0, _v1, _v2, _v3);	
	return vAdd(vAdd(_v0, _v1), vAdd(_v2, _v3));
}

FORCE_INLINE v128 vAddH4x3(v128 _v0, v128 _v1, v128 _v2, v128 _v3)
{
	mTranspose(_v0, _v1, _v2, _v3);
	return vAdd(vAdd(_v0, _v1), _v3);
}

FORCE_INLINE v128 vNormalize(v128 _v)
{
	v128 vLen2 = vAddH(vMul(_v, _v));
	v128 vRcp = vRc(vLen2);
	return vMul(vRcp, _v);
}

FORCE_INLINE void vNormalize4x3(v128 _v0, v128 _v1, v128 _v2, v128 _v3)
{
	mTranspose(_v0, _v1, _v2, _v3);
	v128 vLen2 = vAdd(vAdd(vMul(_v0, _v0), vMul(_v1, _v1)), vAdd(vMul(_v2, _v2), vMul(_v3, _v3)));
	v128 vLenRc = vRsq(vLen2);
	_v0 = vMskW(vMul(vLenRc, _v0), _v0);
	_v1 = vMskW(vMul(vLenRc, _v1), _v1);
	_v2 = vMskW(vMul(vLenRc, _v2), _v2);
	_v3 = vMskW(vMul(vLenRc, _v3), _v3);	
}

FORCE_INLINE void vNormalize4x4(v128 _v0, v128 _v1, v128 _v2, v128 _v3)
{
	mTranspose(_v0, _v1, _v2, _v3);
	v128 vLen2 = vAdd(vAdd(vMul(_v0, _v0), vMul(_v1, _v1)), vMul(_v2, _v2));
	v128 vLenRc = vRsq(vLen2);
	_v0 = vMul(vLenRc, _v0);
	_v1 = vMul(vLenRc, _v1);
	_v2 = vMul(vLenRc, _v2);
	_v3 = vMul(vLenRc, _v3);
}

FORCE_INLINE v128 vNormalize3(v128 _v)
{
	v128 vLen2 = vAddH3(vMul(_v, _v));
	v128 vLenRc = vRsq(vLen2);
	return vMskW(vMul(vLenRc, _v), _v);
}





#endif //INC_SIMD_IMPL_HPP
