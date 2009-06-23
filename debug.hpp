/*
 *	Copyright (c) 2009 Joel de Vahl
 *	              2009 Markus Ålind
 *	see LICENSE file for more info
 */

#ifndef INC_SIMD_DEBUG_HPP
#define INC_SIMD_DEBUG_HPP

#include "intrinsics.hpp"
#include <cstdio>

#define vDebugPrint(V) vDebugPrint_helper(#V, V)
#define vDebugPrint_u32(V) vDebugPrint_u32_helper(#V, V)
#define vDebugPrint_u32x(V) vDebugPrint_u32x_helper(#V, V)

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

#endif
