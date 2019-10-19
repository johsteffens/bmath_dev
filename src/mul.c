/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "mul.h"

void bmath_mf3_s_htp_mul2_htp( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    ASSERT( m->cols == o->rows );
    ASSERT( o->cols == r->rows );
    ASSERT( m->rows == r->cols );

    for( sz_t i = 0; i < r->rows; i++ )
    {
        for( sz_t j = 0; j < r->cols; j++ )
        {
            f3_t sum = 0;
            for( sz_t k = 0; k < o->rows; k++ )
            {
                sum += o->data[ k * o->stride + i ] * m->data[ j * m->stride + k ];
            }
            r->data[ i * r->stride + j ] = sum;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_htp_mul3_htp( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    ASSERT( m->cols == o->rows );
    ASSERT( o->cols == r->rows );
    ASSERT( m->rows == r->cols );

    bmath_vf3_s* ocol = bmath_vf3_s_create();
    bmath_vf3_s_set_size( ocol, o->rows );

    bmath_mf3_s_zro( r );

    for( sz_t i = 0; i < r->rows; i++ )
    {
        for( sz_t k = 0; k < o->rows; k++ )
        {
            ocol->data[ k ] = o->data[ k * o->stride + i ];
        }

        for( sz_t j = 0; j < r->cols; j++ )
        {
            for( sz_t k = 0; k < o->rows; k++ )
            {
                r->data[ i * r->stride + j ] += ocol->data[ k ] * m->data[ j * m->stride + k ];
            }
        }
    }

    bmath_vf3_s_discard( ocol );
}

//----------------------------------------------------------------------------------------------------------------------


