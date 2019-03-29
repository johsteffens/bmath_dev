/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "smul.h"

void bmath_smf3_s_mul1( const bmath_smf3_s* o, const bmath_smf3_s* m, bmath_smf3_s* r )
{
    ASSERT( o->rows           == r->rows );
    ASSERT( o->xons * o->slos == m->rows );
    ASSERT( m->xons           == r->xons );
    ASSERT( m->slos           == r->slos );
    bmath_smf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t o_slo = 0; o_slo < o->slos; o_slo++ )
        {
            for( sz_t o_xon = 0; o_xon < o->xons; o_xon++ )
            {
                f3_t o_val = o->v_data[ o->i_data[ o_row * o->i_stride + o_xon ] + o_slo ];
                sz_t m_row = o_xon * o->slos + o_slo;
                for( sz_t m_xon = 0; m_xon < m->xons; m_xon++ )
                {
                    for( sz_t m_slo = 0; m_slo < m->slos; m_slo++ )
                    {
                        r->v_data[ r->i_data[ o_row * r->i_stride + m_xon ] + m_slo ] +=
                        m->v_data[ m->i_data[ m_row * m->i_stride + m_xon ] + m_slo ] * o_val;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_mul1_eval()
{
    BCORE_LIFE_INIT();

    sz_t slos = 32;
    sz_t rows = slos * 64;
    sz_t xons = 64;
    sz_t cols = xons * slos;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_smf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_smf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_smf3_s, sm3 );

    bmath_smf3_s_set_size( sm1, rows, rows / slos, slos );
    bmath_smf3_s_set_size( sm2, rows, xons, slos );
    bmath_smf3_s_set_size( sm3, rows, xons, slos );

    bmath_mf3_s_set_size( m1, rows, rows );
    bmath_mf3_s_set_size( m2, rows, cols );
    bmath_mf3_s_set_size( m3, rows, cols );
    bmath_mf3_s_set_size( m4, rows, cols );

    u2_t rval = 1234;
    bmath_smf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_smf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_smf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_smf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );
    CPU_TIME_TO_STDOUT( bmath_smf3_s_mul1( sm1, sm2, sm3 ) );

//    bmath_smf3_s_to_stdout( sm3 );

    bmath_smf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_convolution_eval1()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_smf3_s, sm1 );

//    bmath_smf3_s_set_splicing_for_convolution_1d( sm1, 32, 4, 2 );
    bmath_smf3_s_set_splicing_for_convolution_2d( sm1, 3, 3, 2, 3, 1, 1 );
    bmath_smf3_s_fit_size_data( sm1 );
    for( sz_t i = 0; i < sm1->v_size; i++ ) sm1->v_data[ i ] = i;

    bmath_mf3_s_set_size( m1, sm1->rows, sm1->xons * sm1->slos );

    bmath_smf3_s_cpy_ifl_to_mf3( sm1, m1 );

    bmath_mf3_s_to_stdout( m1 );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_convolution_eval2()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_smf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_smf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_smf3_s, sm3 );

    sz_t k_rows  = 32;
    sz_t k_cols  = 32;
    sz_t k_slos  = 16;
    sz_t kernels = 32;
    u2_t rval = 1234;

    bmath_smf3_s_set_splicing_for_convolution_2d( sm1, 64, 64 * k_slos, k_rows, k_cols * k_slos, 1, k_slos );
    bmath_smf3_s_fit_size_data( sm1 );
//    for( sz_t i = 0; i < sm1->v_size; i++ ) sm1->v_data[ i ] = i;
    bmath_smf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
//    bmath_smf3_s_to_stdout( sm1 );

    bmath_smf3_s_set_size( sm2, sm1->xons * sm1->slos, 1, kernels );
    bmath_smf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );

    bmath_smf3_s_set_size( sm3, sm1->rows, 1, kernels );

    bmath_mf3_s_set_size( m1, sm1->rows, sm1->xons * sm1->slos );
    bmath_mf3_s_set_size( m2, sm2->rows, sm2->xons * sm2->slos );
    bmath_mf3_s_set_size( m3, sm3->rows, sm3->xons * sm3->slos );
    bmath_mf3_s_set_size( m4, sm3->rows, sm3->xons * sm3->slos );

    bmath_smf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_smf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_smf3_s_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );

    bmath_smf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}
