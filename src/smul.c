/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "smul.h"

#define BMATH_SMUL_BLOCK_SIZE 32
#define BMATH_SMUL_BLKP4_SIZE ( BMATH_SMUL_BLOCK_SIZE >> 2 )

//----------------------------------------------------------------------------------------------------------------------

/*
static sz_t mf3_sx_midof( sz_t v, const sz_t bz )
{
    return ( v > ( bz << 1 ) ) ? ( v / ( bz << 1 ) ) * bz : bz;
}
*/

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_htp_mul2( const bmath_xsmf3_s* o, const bmath_xsmf3_s* m, bmath_xsmf3_s* r )
{
    ASSERT( o->xons * o->slos == r->rows );
    ASSERT( o->rows           == m->rows );
    ASSERT( m->xons           == r->xons );
    ASSERT( m->slos           == r->slos );
    bmath_xsmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t o_xon = 0; o_xon < o->xons; o_xon++ )
        {
            for( sz_t o_slo = 0; o_slo < o->slos; o_slo++ )
            {
                f3_t o_val = o->v_data[ o->i_data[ o_xon * o->i_stride + o_row ] + o_slo ];
                sz_t m_row = o_row;
                sz_t r_row = o_xon * o->slos + o_slo;
                for( sz_t m_xon = 0; m_xon < m->xons; m_xon++ )
                {
                    for( sz_t m_slo = 0; m_slo < m->slos; m_slo++ )
                    {
                        r->v_data[ r->i_data[ m_xon * r->i_stride + r_row ] + m_slo ] +=
                        m->v_data[ m->i_data[ m_xon * m->i_stride + m_row ] + m_slo ] * o_val;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_htp_mul_eval()
{
    BCORE_LIFE_INIT();

    sz_t rows1 = 2357;
    sz_t xons1 = 27;
    sz_t slos1 = 64;

    sz_t rows2 = rows1;
    sz_t xons2 = 61;
    sz_t slos2 = 32;

    sz_t rows3 = xons1 * slos1;
    sz_t xons3 = xons2;
    sz_t slos3 = slos2;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm3 );

    bmath_xsmf3_s_set_size( sm1, rows1, xons1, slos1 );
    bmath_xsmf3_s_set_size( sm2, rows2, xons2, slos2 );
    bmath_xsmf3_s_set_size( sm3, rows3, xons3, slos3 );

    bmath_mf3_s_set_size( m1, rows1, xons1 * slos1 );
    bmath_mf3_s_set_size( m2, rows2, xons2 * slos2 );
    bmath_mf3_s_set_size( m3, rows3, xons3 * slos3 );
    bmath_mf3_s_set_size( m4, rows3, xons3 * slos3 );

    u2_t rval = 1234;
    bmath_xsmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_htp_mul( m1, m2, m3 ) );
//    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_htp_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_htp_mul( sm1, sm2, sm3 ) );

//    bmath_xsmf3_s_to_stdout( sm3 );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_mul_htp1( const bmath_xsmf3_s* o, const bmath_xsmf3_s* m, bmath_xsmf3_s* r )
{
    ASSERT( o->rows == r->rows );
    ASSERT( m->rows == r->xons * r->slos );
    ASSERT( o->xons == m->xons );
    ASSERT( o->slos == m->slos );
    bmath_xsmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t r_xon = 0; r_xon < r->xons; r_xon++ )
        {
            for( sz_t r_slo = 0; r_slo < r->slos; r_slo++ )
            {
                f3_t* dst = &r->v_data[ r->i_data[ r_xon * r->i_stride + o_row ] + r_slo ];
                sz_t m_row = r_xon * r->slos + r_slo;
                for( sz_t o_xon = 0; o_xon < o->xons; o_xon++ )
                {
                    for( sz_t o_slo = 0; o_slo < o->slos; o_slo++ )
                    {
                        *dst += o->v_data[ o->i_data[ o_xon * o->i_stride + o_row ] + o_slo ] *
                                m->v_data[ m->i_data[ o_xon * m->i_stride + m_row ] + o_slo ];
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_mul_htp_eval()
{
    BCORE_LIFE_INIT();

    sz_t rows1 = 1024;
    sz_t xons1 = 32;
    sz_t slos1 = 64;

    sz_t rows3 = rows1;
    sz_t xons3 = 32;
    sz_t slos3 = 64;

    sz_t rows2 = xons3 * slos3;
    sz_t xons2 = xons1;
    sz_t slos2 = slos1;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm3 );

    bmath_xsmf3_s_set_size( sm1, rows1, xons1, slos1 );
    bmath_xsmf3_s_set_size( sm2, rows2, xons2, slos2 );
    bmath_xsmf3_s_set_size( sm3, rows3, xons3, slos3 );

    bmath_mf3_s_set_size( m1, rows1, xons1 * slos1 );
    bmath_mf3_s_set_size( m2, rows2, xons2 * slos2 );
    bmath_mf3_s_set_size( m3, rows3, xons3 * slos3 );
    bmath_mf3_s_set_size( m4, rows3, xons3 * slos3 );

    u2_t rval = 1234;
    bmath_xsmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul_htp( m1, m2, m3 ) );
    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_mul_htp(  sm1, sm2, sm3 ) );

//    bmath_xsmf3_s_to_stdout( sm3 );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_mul1( const bmath_xsmf3_s* o, const bmath_xsmf3_s* m, bmath_xsmf3_s* r )
{
    ASSERT( o->rows           == r->rows );
    ASSERT( o->xons * o->slos == m->rows );
    ASSERT( m->xons           == r->xons );
    ASSERT( m->slos           == r->slos );
    bmath_xsmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t o_xon = 0; o_xon < o->xons; o_xon++ )
        {
            for( sz_t o_slo = 0; o_slo < o->slos; o_slo++ )
            {
                f3_t o_val = o->v_data[ o->i_data[ o_xon * o->i_stride + o_row ] + o_slo ];
                sz_t m_row = o_xon * o->slos + o_slo;
                for( sz_t m_xon = 0; m_xon < m->xons; m_xon++ )
                {
                    for( sz_t m_slo = 0; m_slo < m->slos; m_slo++ )
                    {
                        r->v_data[ r->i_data[ m_xon * r->i_stride + o_row ] + m_slo ] +=
                        m->v_data[ m->i_data[ m_xon * m->i_stride + m_row ] + m_slo ] * o_val;
                    }
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_mul_eval()
{
    BCORE_LIFE_INIT();

    sz_t rows1 = 4096;
    sz_t xons1 = 32;
    sz_t slos1 = 32;
    sz_t xons2 = 32;
    sz_t slos2 = 64;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm3 );

    bmath_xsmf3_s_set_size( sm1, rows1,         xons1, slos1 );
    bmath_xsmf3_s_set_size( sm2, xons1 * slos1, xons2, slos2 );
    bmath_xsmf3_s_set_size( sm3, rows1,         xons2, slos2 );

    bmath_mf3_s_set_size( m1, rows1,         xons1 * slos1 );
    bmath_mf3_s_set_size( m2, xons1 * slos1, xons2 * slos2 );
    bmath_mf3_s_set_size( m3, rows1,         xons2 * slos2 );
    bmath_mf3_s_set_size( m4, rows1,         xons2 * slos2 );

    u2_t rval = 1234;
    bmath_xsmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );
//    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_mul(  sm1, sm2, sm3 ) );

//    bmath_xsmf3_s_to_stdout( sm3 );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

/**********************************************************************************************************************/

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_htp_mul1( const bmath_asmf3_s* o, const bmath_asmf3_s* m, bmath_asmf3_s* r )
{
    ASSERT( o->cols == r->rows );
    ASSERT( o->rows == m->rows );
    ASSERT( m->cols == r->cols );
    bmath_asmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t o_col = 0; o_col < o->cols; o_col++ )
        {
            f3_t o_val = o->v_data[ o->i_data[ o_row * o->i_stride + o_col ] ];
            sz_t m_row = o_row;
            sz_t r_row = o_col;
            for( sz_t m_col = 0; m_col < m->cols; m_col++ )
            {
                r->v_data[ r->i_data[ r_row * r->i_stride + m_col ] ] +=
                m->v_data[ m->i_data[ m_row * m->i_stride + m_col ] ] * o_val;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_htp_mul_eval()
{
    BCORE_LIFE_INIT();

    sz_t rows1 = 2357;
    sz_t cols1 = 1076;
    sz_t cols2 = 1998;

    sz_t rows2 = rows1;
    sz_t rows3 = cols1;
    sz_t cols3 = cols2;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_asmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm3 );

    bmath_asmf3_s_set_size( sm1, rows1, cols1 );
    bmath_asmf3_s_set_size( sm2, rows2, cols2 );
    bmath_asmf3_s_set_size( sm3, rows3, cols3 );

    bmath_mf3_s_set_size( m1, rows1, cols1 );
    bmath_mf3_s_set_size( m2, rows2, cols2 );
    bmath_mf3_s_set_size( m3, rows3, cols3 );
    bmath_mf3_s_set_size( m4, rows3, cols3 );

    u2_t rval = 1234;
    bmath_asmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_htp_mul( m1, m2, m3 ) );
//    CPU_TIME_TO_STDOUT( bmath_asmf3_s_htp_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_asmf3_s_htp_mul( sm1, sm2, sm3 ) );

//    bmath_asmf3_s_to_stdout( sm3 );

    bmath_asmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_mul_htp1( const bmath_asmf3_s* o, const bmath_asmf3_s* m, bmath_asmf3_s* r )
{
    ASSERT( o->rows == r->rows );
    ASSERT( m->rows == r->cols );
    ASSERT( o->cols == m->cols );
    bmath_asmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t r_col = 0; r_col < r->cols; r_col++ )
        {
            f3_t* dst = &r->v_data[ r->i_data[ o_row * r->i_stride + r_col ] ];
            sz_t m_row = r_col;
            for( sz_t o_col = 0; o_col < o->cols; o_col++ )
            {
                *dst += o->v_data[ o->i_data[ o_row * o->i_stride + o_col ] ] *
                        m->v_data[ m->i_data[ m_row * m->i_stride + o_col ] ];
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_mul_htp_eval()
{
    BCORE_LIFE_INIT();

//    sz_t rows1 = 1024;
//    sz_t cols1 = 32 * 64;
//    sz_t cols3 = 32 * 64;

    sz_t rows1 = 1347;
    sz_t cols1 = 2153;
    sz_t cols3 = 1977;

    sz_t rows3 = rows1;
    sz_t rows2 = cols3;
    sz_t cols2 = cols1;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_asmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm3 );

    bmath_asmf3_s_set_size( sm1, rows1, cols1 );
    bmath_asmf3_s_set_size( sm2, rows2, cols2 );
    bmath_asmf3_s_set_size( sm3, rows3, cols3 );

    bmath_mf3_s_set_size( m1, rows1, cols1 );
    bmath_mf3_s_set_size( m2, rows2, cols2 );
    bmath_mf3_s_set_size( m3, rows3, cols3 );
    bmath_mf3_s_set_size( m4, rows3, cols3 );

    u2_t rval = 1234;
    bmath_asmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul_htp( m1, m2, m3 ) );
//    CPU_TIME_TO_STDOUT( bmath_asmf3_s_mul_htp1(  sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_asmf3_s_mul_htp(  sm1, sm2, sm3 ) );

//    bmath_asmf3_s_to_stdout( sm3 );

    bmath_asmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_mul1( const bmath_asmf3_s* o, const bmath_asmf3_s* m, bmath_asmf3_s* r )
{
    ASSERT( o->rows == r->rows );
    ASSERT( o->cols == m->rows );
    ASSERT( m->cols == r->cols );
    bmath_asmf3_s_zro( r );

    for( sz_t o_row = 0; o_row < o->rows; o_row++ )
    {
        for( sz_t o_col = 0; o_col < o->cols; o_col++ )
        {
            f3_t o_val = o->v_data[ o->i_data[ o_row * o->i_stride + o_col ] ];
            sz_t m_row = o_col;
            for( sz_t m_col = 0; m_col < m->cols; m_col++ )
            {
                r->v_data[ r->i_data[ o_row * r->i_stride + m_col ] ] +=
                m->v_data[ m->i_data[ m_row * m->i_stride + m_col ] ] * o_val;
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_mul_eval()
{
    BCORE_LIFE_INIT();

    sz_t rows1 = 4096;
    sz_t cols1 = 1024;
    sz_t cols2 = 2048;

//    sz_t rows1 = 3657;
//    sz_t cols1 = 1234;
//    sz_t cols2 = 1768;

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_asmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm3 );

    bmath_asmf3_s_set_size( sm1, rows1, cols1 );
    bmath_asmf3_s_set_size( sm2, cols1, cols2 );
    bmath_asmf3_s_set_size( sm3, rows1, cols2 );

    bmath_mf3_s_set_size( m1, rows1, cols1 );
    bmath_mf3_s_set_size( m2, cols1, cols2 );
    bmath_mf3_s_set_size( m3, rows1, cols2 );
    bmath_mf3_s_set_size( m4, rows1, cols2 );

    u2_t rval = 1234;
    bmath_asmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_asmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );
//    CPU_TIME_TO_STDOUT( bmath_asmf3_s_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_asmf3_s_mul(  sm1, sm2, sm3 ) );

//    bmath_asmf3_s_to_stdout( sm3 );

    bmath_asmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_convolution_eval1()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm1 );

    bmath_xsmf3_s_set_splicing_for_convolution_1d( sm1, 32, 4, 2 );
//    bmath_xsmf3_s_set_splicing_for_convolution_2d( sm1, 3, 3, 2, 3, 1, 1 );
    bmath_xsmf3_s_fit_size_data( sm1 );
    for( sz_t i = 0; i < sm1->v_size; i++ ) sm1->v_data[ i ] = i;

    bmath_mf3_s_set_size( m1, sm1->rows, sm1->xons * sm1->slos );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm1, m1 );

    bmath_mf3_s_to_stdout( m1 );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_asmf3_s_convolution_eval1()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_asmf3_s, sm1 );

    bmath_asmf3_s_set_splicing_for_convolution_1d( sm1, 32, 4, 2 );
//    bmath_asmf3_s_set_splicing_for_convolution_2d( sm1, 3, 3, 2, 3, 1, 1 );
    bmath_asmf3_s_fit_size_data( sm1 );
    for( sz_t i = 0; i < sm1->v_size; i++ ) sm1->v_data[ i ] = i;

    bmath_mf3_s_set_size( m1, sm1->rows, sm1->cols );

    bmath_asmf3_s_cpy_ifl_to_mf3( sm1, m1 );

    bmath_mf3_s_to_stdout( m1 );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_xsmf3_s_convolution_eval2()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m4 );

    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm1 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm2 );
    BCORE_LIFE_CREATE( bmath_xsmf3_s, sm3 );

    sz_t k_rows  = 32;
    sz_t k_cols  = 32;
    sz_t k_slos  = 16;
    sz_t kernels = 32;
    u2_t rval = 1234;

    bmath_xsmf3_s_set_splicing_for_convolution_2d( sm1, 64, 64 * k_slos, k_rows, k_cols * k_slos, 1, k_slos );
    bmath_xsmf3_s_fit_size_data( sm1 );
//    for( sz_t i = 0; i < sm1->v_size; i++ ) sm1->v_data[ i ] = i;
    bmath_xsmf3_s_set_random( sm1, false, false, 0, 1.0, -1.0, 1.0, &rval );
//    bmath_xsmf3_s_to_stdout( sm1 );

    bmath_xsmf3_s_set_size( sm2, sm1->xons * sm1->slos, 1, kernels );
    bmath_xsmf3_s_set_random( sm2, false, false, 0, 1.0, -1.0, 1.0, &rval );

    bmath_xsmf3_s_set_size( sm3, sm1->rows, 1, kernels );

    bmath_mf3_s_set_size( m1, sm1->rows, sm1->xons * sm1->slos );
    bmath_mf3_s_set_size( m2, sm2->rows, sm2->xons * sm2->slos );
    bmath_mf3_s_set_size( m3, sm3->rows, sm3->xons * sm3->slos );
    bmath_mf3_s_set_size( m4, sm3->rows, sm3->xons * sm3->slos );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm1, m1 );
    bmath_xsmf3_s_cpy_ifl_to_mf3( sm2, m2 );

    CPU_TIME_TO_STDOUT( bmath_xsmf3_s_mul( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );

    bmath_xsmf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}
