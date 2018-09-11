/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#ifndef ZORDER_H
#define ZORDER_H

#include "bmath_std.h"
#include "bmath_mf3.h"

void bmath_zorder_fwd_cpy( const f3_t* o, sz_t o_stride, sz_t o_rows,       f3_t* r ); // o -> r
void bmath_zorder_rev_cpy(       f3_t* o, sz_t o_stride, sz_t o_rows, const f3_t* r ); // o <- r

void bmath_mf3_s_zorder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r );
void bmath_mf3_s_norder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r );
void bmath_mf3_s_morder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r );

void bmath_zorder_test( void );

#endif  // ZORDER_H
