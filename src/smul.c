/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "smul.h"

#define BMATH_SMUL_BLOCK_SIZE 32

static sz_t smf3_midof( sz_t v, const sz_t bz )
{
    return ( v > ( bz << 1 ) ) ? ( v / ( bz << 1 ) ) * bz : bz;
}

//----------------------------------------------------------------------------------------------------------------------

#ifdef BMATH_AVX
/// smul: Fixed size AVX-Microkernel
static void bmath_smf3_s_mul_rsblock_avx_fix_kernel
(
    const bmath_smf3_s* o, sz_t o_row, sz_t o_xon, sz_t o_slo,
    const bmath_smf3_s* m,             sz_t m_xon, sz_t m_slo,
    bmath_smf3_s* r
)
{
    assert( ( BMATH_SMUL_BLOCK_SIZE & 3 ) == 0 );

    #define BMATH_SMUL_BLKP4_SIZE ( BMATH_SMUL_BLOCK_SIZE >> 2 )

    __m256d r_p4[ BMATH_SMUL_BLKP4_SIZE ];
    __m256d m_p4[ BMATH_SMUL_BLOCK_SIZE ][ BMATH_SMUL_BLKP4_SIZE ];

    for( sz_t j = 0; j < BMATH_SMUL_BLOCK_SIZE; j++ )
    {
        for( sz_t k = 0; k < BMATH_SMUL_BLKP4_SIZE; k++ )
        {
            m_p4[ j ][ k ] = _mm256_loadu_pd( &m->v_data[ m->i_data[ m_xon * m->i_stride + ( o_xon * o->slos + o_slo + j ) ] + m_slo + k * 4 ] );
        }
    }

    for( sz_t i = 0; i < BMATH_SMUL_BLOCK_SIZE; i++ )
    {
        const f3_t* o_d = &o->v_data[ o->i_data[ o_xon * o->i_stride + ( o_row + i ) ] + o_slo ];
              f3_t* r_d = &r->v_data[ r->i_data[ m_xon * r->i_stride + ( o_row + i ) ] + m_slo ];

        __m256d o_p4 = _mm256_set1_pd( o_d[ 0 ] );

        for( sz_t k = 0; k < BMATH_SMUL_BLKP4_SIZE; k++ )
        {
            r_p4[ k ] = _mm256_mul_pd( m_p4[ 0 ][ k ], o_p4 );
        }

        for( sz_t j = 1; j < BMATH_SMUL_BLOCK_SIZE; j++ )
        {
            o_p4 = _mm256_set1_pd( o_d[ j ] );
            for( sz_t k = 0; k < BMATH_SMUL_BLKP4_SIZE; k++ )
            {
                #ifdef BMATH_AVX2_FMA
                    r_p4[ k ] = _mm256_fmadd_pd( m_p4[ j ][ k ], o_p4, r_p4[ k ] );
                #else
                    r_p4[ k ] = _mm256_add_pd( _mm256_mul_pd( m_p4[ j ][ k ], o_p4 ), r_p4[ k ] );
                #endif // BMATH_AVX2_FMA
            }
        }

        for( sz_t k = 0; k < BMATH_SMUL_BLKP4_SIZE; k++ )
        {
            _mm256_storeu_pd( r_d + k * 4, _mm256_add_pd( _mm256_loadu_pd( r_d + k * 4 ), r_p4[ k ] ) );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

/** smul: Flexible AVX-Microkernel
 *  Allows all combinations o_rows, o_slos, m_slos (including 0)
 *  provided o_rows <= BMATH_SMUL_BLOCK_SIZE && m_slos <= BMATH_SMUL_BLOCK_SIZE
 */
static void bmath_smf3_s_mul_rsblock_avx_flex_kernel
(
    const bmath_smf3_s* o, sz_t o_row, sz_t o_rows, sz_t o_xon, sz_t o_slo, sz_t o_slos,
    const bmath_smf3_s* m,                          sz_t m_xon, sz_t m_slo, sz_t m_slos,
    bmath_smf3_s* r
)
{
    #define BMATH_SMUL_BLKP4_SIZE ( BMATH_SMUL_BLOCK_SIZE >> 2 )

    ASSERT( o_slos <= BMATH_SMUL_BLOCK_SIZE );
    ASSERT( m_slos <= BMATH_SMUL_BLOCK_SIZE );

    const sz_t m_slos4l = m_slos >> 2;
    const sz_t m_slosr  = m_slos - m_slos4l * 4;
    const sz_t m_slos4p = m_slosr > 0 ? m_slos4l + 1 : m_slos4l;

    __m256d r_p4[ BMATH_SMUL_BLKP4_SIZE ];
    __m256d m_p4[ BMATH_SMUL_BLOCK_SIZE ][ BMATH_SMUL_BLKP4_SIZE ];

    for( sz_t j = 0; j < o_slos; j++ )
    {
        for( sz_t k = 0; k < m_slos4l; k++ )
        {
            m_p4[ j ][ k ] = _mm256_loadu_pd( &m->v_data[ m->i_data[ m_xon * m->i_stride + ( o_xon * o->slos + o_slo + j ) ] + m_slo + k * 4 ] );
        }

        if( m_slosr > 0 )
        {
            m_p4[ j ][ m_slos4l ] = _mm256_set1_pd( 0 );
            for( sz_t k = 0; k < m_slosr;  k++ )
            {
                m_p4[ j ][ m_slos4l ][ k ] = m->v_data[ m->i_data[ m_xon * m->i_stride + ( o_xon * o->slos + o_slo + j ) ] + m_slo + m_slos4l * 4 + k ];
            }
        }
    }

    for( sz_t i = 0; i < o_rows; i++ )
    {
        const f3_t* o_d = &o->v_data[ o->i_data[ o_xon * o->i_stride + ( o_row + i ) ] + o_slo ];
              f3_t* r_d = &r->v_data[ r->i_data[ m_xon * r->i_stride + ( o_row + i ) ] + m_slo ];

        for( sz_t k = 0; k < m_slos4p; k++ ) r_p4[ k ] = _mm256_set1_pd( 0 );

        for( sz_t j = 0; j < o_slos; j++ )
        {
            __m256d o_p4 = _mm256_set1_pd( o_d[ j ] );
            for( sz_t k = 0; k < m_slos4p; k++ )
            {
                #ifdef BMATH_AVX2_FMA
                    r_p4[ k ] = _mm256_fmadd_pd( m_p4[ j ][ k ], o_p4, r_p4[ k ] );
                #else
                    r_p4[ k ] = _mm256_add_pd( _mm256_mul_pd( m_p4[ j ][ k ], o_p4 ), r_p4[ k ] );
                #endif // BMATH_AVX2_FMA
            }
        }

        for( sz_t k = 0; k < m_slos4l; k++ ) _mm256_storeu_pd( r_d + k * 4, _mm256_add_pd( _mm256_loadu_pd( r_d + k * 4 ), r_p4[ k ] ) );

        if( m_slosr > 0 )
        {
            for( sz_t k = 0; k < m_slosr; k++ ) r_d[ m_slos4l * 4 + k ] += r_p4[ m_slos4l ][ k ];
        }
    }
}

#endif // BMATH_AVX

//----------------------------------------------------------------------------------------------------------------------

/// xons == 1
void bmath_smf3_s_mul_rsblock
(
    const bmath_smf3_s* o, sz_t o_row, sz_t o_rows, sz_t o_xon, sz_t o_slo, sz_t o_slos,
    const bmath_smf3_s* m,                          sz_t m_xon, sz_t m_slo, sz_t m_slos,
    bmath_smf3_s* r
)
{
    if( o_rows == BMATH_SMUL_BLOCK_SIZE && o_slos == BMATH_SMUL_BLOCK_SIZE && m_slos == BMATH_SMUL_BLOCK_SIZE )
    {
        #ifdef BMATH_AVX
            bmath_smf3_s_mul_rsblock_avx_fix_kernel( o, o_row, o_xon, o_slo, m, m_xon, m_slo, r );
        #else
            f3_t r_p[ BMATH_SMUL_BLOCK_SIZE ];
            f3_t m_p[ BMATH_SMUL_BLOCK_SIZE ][ BMATH_SMUL_BLOCK_SIZE ];
            for( sz_t j = 0; j < BMATH_SMUL_BLOCK_SIZE; j++ )
            {
                for( sz_t k = 0; k < BMATH_SMUL_BLOCK_SIZE; k++ ) m_p[ j ][ k ] = m->v_data[ m->i_data[ m_xon * m->i_stride + ( o_xon * o->slos + o_slo + j ) ] + m_slo + k ];
            }

            for( sz_t i = 0; i < BMATH_SMUL_BLOCK_SIZE; i++ )
            {
                const f3_t* o_d = &o->v_data[ o->i_data[ o_xon * o->i_stride + ( o_row + i ) ] + o_slo ];
                for( sz_t k = 0; k < BMATH_SMUL_BLOCK_SIZE; k++ ) r_p[ k ] = 0;

                for( sz_t j = 0; j < BMATH_SMUL_BLOCK_SIZE; j++ )
                {
                    for( sz_t k = 0; k < BMATH_SMUL_BLOCK_SIZE; k++ ) r_p[ k ] += m_p[ j ][ k ] * o_d[ j ];
                }

                f3_t* r_d = &r->v_data[ r->i_data[ m_xon * r->i_stride + ( o_row + i ) ] + m_slo ];
                for( sz_t k = 0; k < BMATH_SMUL_BLOCK_SIZE; k++ ) r_d[ k ] += r_p[ k ];
            }
        #endif // BMATH_AVX
        return;
    }

    if( o_rows >= o_slos && o_rows >= m_slos && o_rows > BMATH_SMUL_BLOCK_SIZE )
    {
        sz_t mid = smf3_midof( o_rows, BMATH_SMUL_BLOCK_SIZE );
        bmath_smf3_s_mul_rsblock( o, o_row,                mid, o_xon, o_slo, o_slos, m, m_xon, m_slo, m_slos, r );
        bmath_smf3_s_mul_rsblock( o, o_row + mid, o_rows - mid, o_xon, o_slo, o_slos, m, m_xon, m_slo, m_slos, r );
        return;
    }

    if( o_slos >= m_slos && o_slos > BMATH_SMUL_BLOCK_SIZE )
    {
        sz_t mid = smf3_midof( o_slos, BMATH_SMUL_BLOCK_SIZE );
        bmath_smf3_s_mul_rsblock( o, o_row, o_rows, o_xon, o_slo,                mid, m, m_xon, m_slo, m_slos, r );
        bmath_smf3_s_mul_rsblock( o, o_row, o_rows, o_xon, o_slo + mid, o_slos - mid, m, m_xon, m_slo, m_slos, r );
        return;
    }

    if( m_slos > BMATH_SMUL_BLOCK_SIZE )
    {
        sz_t mid = smf3_midof( m_slos, BMATH_SMUL_BLOCK_SIZE );
        bmath_smf3_s_mul_rsblock( o, o_row, o_rows, o_xon, o_slo, o_slos, m, m_xon, m_slo,                mid, r );
        bmath_smf3_s_mul_rsblock( o, o_row, o_rows, o_xon, o_slo, o_slos, m, m_xon, m_slo + mid, m_slos - mid, r );
        return;
    }

    #ifdef BMATH_AVX
        bmath_smf3_s_mul_rsblock_avx_flex_kernel( o, o_row, o_rows, o_xon, o_slo, o_slos, m, m_xon, m_slo, m_slos, r );
    #else
        f3_t r_p[ BMATH_SMUL_BLOCK_SIZE ];
        f3_t m_p[ BMATH_SMUL_BLOCK_SIZE ][ BMATH_SMUL_BLOCK_SIZE ];
        for( sz_t j = 0; j < o_slos; j++ )
        {
            for( sz_t k = 0; k < m_slos; k++ ) m_p[ j ][ k ] = m->v_data[ m->i_data[ m_xon * m->i_stride + ( o_xon * o->slos + o_slo + j ) ] + m_slo + k ];
        }

        for( sz_t i = 0; i < o_rows; i++ )
        {
            const f3_t* o_d = &o->v_data[ o->i_data[ o_xon * o->i_stride + ( o_row + i ) ] + o_slo ];
            for( sz_t k = 0; k < m_slos; k++ ) r_p[ k ] = 0;

            for( sz_t j = 0; j < o_slos; j++ )
            {
                for( sz_t k = 0; k < m_slos; k++ ) r_p[ k ] += m_p[ j ][ k ] * o_d[ j ];
            }

            f3_t* r_d = &r->v_data[ r->i_data[ m_xon * r->i_stride + ( o_row + i ) ] + m_slo ];
            for( sz_t k = 0; k < m_slos; k++ ) r_d[ k ] += r_p[ k ];
        }
    #endif // BMATH_AVX
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_mul_rxblock
(
    const bmath_smf3_s* o, sz_t o_row, sz_t o_rows, sz_t o_xon, sz_t o_xons,
    const bmath_smf3_s* m,                          sz_t m_xon, sz_t m_xons,
          bmath_smf3_s* r
)
{
    if( o_rows >= o_xons * o->slos && o_rows >= m_xons * m->slos && o_rows > BMATH_SMUL_BLOCK_SIZE )
    {
        sz_t mid = smf3_midof( o_rows, BMATH_SMUL_BLOCK_SIZE );
        bmath_smf3_s_mul_rxblock( o, o_row,                mid, o_xon, o_xons, m, m_xon, m_xons, r );
        bmath_smf3_s_mul_rxblock( o, o_row + mid, o_rows - mid, o_xon, o_xons, m, m_xon, m_xons, r );
        return;
    }

    if( o_xons >= m_xons && o_xons > 1 )
    {
        sz_t mid = smf3_midof( o_xons, 1 );
        bmath_smf3_s_mul_rxblock( o, o_row, o_rows, o_xon,                mid, m, m_xon, m_xons, r );
        bmath_smf3_s_mul_rxblock( o, o_row, o_rows, o_xon + mid, o_xons - mid, m, m_xon, m_xons, r );
        return;
    }

    if( m_xons > 1 )
    {
        sz_t mid = smf3_midof( m_xons, 1 );
        bmath_smf3_s_mul_rxblock( o, o_row, o_rows, o_xon, o_xons, m, m_xon,                mid, r );
        bmath_smf3_s_mul_rxblock( o, o_row, o_rows, o_xon, o_xons, m, m_xon + mid, m_xons - mid, r );
        return;
    }

    bmath_smf3_s_mul_rsblock( o, o_row, o_rows, o_xon, 0, o->slos, m, m_xon, 0, m->slos, r );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_mul2( const bmath_smf3_s* o, const bmath_smf3_s* m, bmath_smf3_s* r )
{
    ASSERT( o->rows           == r->rows );
    ASSERT( o->xons * o->slos == m->rows );
    ASSERT( m->xons           == r->xons );
    ASSERT( m->slos           == r->slos );
    bmath_smf3_s_zro( r );
    bmath_smf3_s_mul_rxblock( o, 0, o->rows, 0, o->xons, m, 0, m->xons, r );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_smf3_s_mul1( const bmath_smf3_s* o, const bmath_smf3_s* m, bmath_smf3_s* r )
{
    ASSERT( o->rows           == r->rows );
    ASSERT( o->xons * o->slos == m->rows );
    ASSERT( m->xons           == r->xons );
    ASSERT( m->slos           == r->slos );
    bmath_smf3_s_zro( r );

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

void bmath_smf3_s_mul_eval()
{
    BCORE_LIFE_INIT();

    sz_t slos = 31;
    sz_t rows = slos * 64;
    sz_t xons = 32;
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
//    CPU_TIME_TO_STDOUT( bmath_smf3_s_mul1( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_smf3_s_mul2( sm1, sm2, sm3 ) );

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

    CPU_TIME_TO_STDOUT( bmath_smf3_s_mul2( sm1, sm2, sm3 ) );
    CPU_TIME_TO_STDOUT( bmath_mf3_s_mul( m1, m2, m3 ) );

    bmath_smf3_s_cpy_ifl_to_mf3( sm3, m4 );

    ASSERT( bmath_mf3_s_is_near_equ( m3, m4, 1E-8 ) );

    BCORE_LIFE_DOWN();
}
