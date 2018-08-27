/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#ifndef SVD_H
#define SVD_H

#include "bmath_std.h"
#include "bmath_mf3.h"

void bmath_mf3_s_ubd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v ); // upper bidiagonal
void bmath_mf3_s_lbd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v ); // lower bidiagonal

void bmath_mf3_s_qrd4( bmath_mf3_s* u, bmath_mf3_s* a );
void bmath_mf3_s_qrd_pmt4( bmath_mf3_s* u, bmath_mf3_s* a, bmath_pmt_s* p );

void bmath_mf3_s_qrd_pmt2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_pmt_s* p );

void bmath_mf3_s_trd2( bmath_mf3_s* a, bmath_mf3_s* v );

void bmath_mf3_s_decompose_qrd5( bmath_mf3_s* u, bmath_mf3_s* a );
void bmath_mf3_s_decompose_lqd5( bmath_mf3_s* a, bmath_mf3_s* v );

void bmath_mf3_s_decompose_ubd5( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v );
void bmath_mf3_s_decompose_lbd5( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v );

/// older code
bl_t bmath_mf3_s_svd_full( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v );
bl_t bmath_mf3_s_svd2(  bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v );


#endif  // SVD_H
