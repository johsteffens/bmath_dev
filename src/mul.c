/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "mul.h"

//---------------------------------------------------------------------------------------------------------------------

static const sz_t mul_kernel_o_rows = 4 * 8;
static const sz_t mul_kernel_o_cols = 4 * 8;
static const sz_t mul_kernel_m_cols = 4 * 8;

static void bmath_mf3_s_mul_avx_microkernel_01( const f3_t* o, sz_t o_s, const f3_t* m, sz_t m_s, f3_t* r, sz_t r_s )
{
    assert( mul_kernel_m_cols == 32 );
    __m256d r_p4[ 8 ];
    for( sz_t i = 0; i < mul_kernel_o_rows; i++ )
    {
        const f3_t* oi = o + i * o_s;
              f3_t* ri = r + i * r_s;

        r_p4[ 0 ] = _mm256_loadu_pd( ri + 0 * 4 );
        r_p4[ 1 ] = _mm256_loadu_pd( ri + 1 * 4 );
        r_p4[ 2 ] = _mm256_loadu_pd( ri + 2 * 4 );
        r_p4[ 3 ] = _mm256_loadu_pd( ri + 3 * 4 );
        r_p4[ 4 ] = _mm256_loadu_pd( ri + 4 * 4 );
        r_p4[ 5 ] = _mm256_loadu_pd( ri + 5 * 4 );
        r_p4[ 6 ] = _mm256_loadu_pd( ri + 6 * 4 );
        r_p4[ 7 ] = _mm256_loadu_pd( ri + 7 * 4 );

        for( sz_t j = 0; j < mul_kernel_o_cols; j++ )
        {
            __m256d o_p4 = _mm256_set1_pd( oi[ j ] );

            const f3_t* mj = m + j * m_s;
            r_p4[ 0 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 0 * 4 ), o_p4, r_p4[ 0 ] );
            r_p4[ 1 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 1 * 4 ), o_p4, r_p4[ 1 ] );
            r_p4[ 2 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 2 * 4 ), o_p4, r_p4[ 2 ] );
            r_p4[ 3 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 3 * 4 ), o_p4, r_p4[ 3 ] );
            r_p4[ 4 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 4 * 4 ), o_p4, r_p4[ 4 ] );
            r_p4[ 5 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 5 * 4 ), o_p4, r_p4[ 5 ] );
            r_p4[ 6 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 6 * 4 ), o_p4, r_p4[ 6 ] );
            r_p4[ 7 ] = _mm256_fmadd_pd( _mm256_loadu_pd( mj + 7 * 4 ), o_p4, r_p4[ 7 ] );
        }

        _mm256_storeu_pd( ri + 0 * 4, r_p4[ 0 ] );
        _mm256_storeu_pd( ri + 1 * 4, r_p4[ 1 ] );
        _mm256_storeu_pd( ri + 2 * 4, r_p4[ 2 ] );
        _mm256_storeu_pd( ri + 3 * 4, r_p4[ 3 ] );
        _mm256_storeu_pd( ri + 4 * 4, r_p4[ 4 ] );
        _mm256_storeu_pd( ri + 5 * 4, r_p4[ 5 ] );
        _mm256_storeu_pd( ri + 6 * 4, r_p4[ 6 ] );
        _mm256_storeu_pd( ri + 7 * 4, r_p4[ 7 ] );
    }
}

static void bmath_mf3_s_mul_avx_microkernel_02( const f3_t* o, sz_t o_s, const f3_t* m, sz_t m_s, f3_t* r, sz_t r_s )
{
    assert( mul_kernel_o_rows == 32 );
    assert( mul_kernel_o_cols == 32 );
    assert( mul_kernel_m_cols == 32 );

    for( sz_t k = 0; k < mul_kernel_o_cols; k += 4 )
    {

        for( sz_t i = 0; i < mul_kernel_o_rows; i += 4 )
        {
            const f3_t* ob = o + i * o_s + k;

            __m256d o_p4[ 4 ];
            o_p4[ 0 ] = _mm256_loadu_pd( ob + 0 * o_s );
            o_p4[ 1 ] = _mm256_loadu_pd( ob + 1 * o_s );
            o_p4[ 2 ] = _mm256_loadu_pd( ob + 2 * o_s );
            o_p4[ 3 ] = _mm256_loadu_pd( ob + 3 * o_s );

            for( sz_t j = 0; j < mul_kernel_o_cols; j += 4 )
            {
                const f3_t* mb = m + k * m_s + j;
                      f3_t* rb = r + i * r_s + j;

                __m256d m_p4[ 4 ];
                m_p4[ 0 ] = _mm256_loadu_pd( mb + 0 * m_s );
                m_p4[ 1 ] = _mm256_loadu_pd( mb + 1 * m_s );
                m_p4[ 2 ] = _mm256_loadu_pd( mb + 2 * m_s );
                m_p4[ 3 ] = _mm256_loadu_pd( mb + 3 * m_s );

                __m256d r_p4;

                r_p4 = _mm256_loadu_pd( rb + 0 * r_s );
                r_p4 = _mm256_fmadd_pd( m_p4[ 0 ], _mm256_set1_pd( o_p4[ 0 ][ 0 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 1 ], _mm256_set1_pd( o_p4[ 0 ][ 1 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 2 ], _mm256_set1_pd( o_p4[ 0 ][ 2 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 3 ], _mm256_set1_pd( o_p4[ 0 ][ 3 ] ), r_p4 );
                _mm256_storeu_pd( rb + 0 * r_s, r_p4 );

                r_p4 = _mm256_loadu_pd( rb + 1 * r_s );
                r_p4 = _mm256_fmadd_pd( m_p4[ 0 ], _mm256_set1_pd( o_p4[ 1 ][ 0 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 1 ], _mm256_set1_pd( o_p4[ 1 ][ 1 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 2 ], _mm256_set1_pd( o_p4[ 1 ][ 2 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 3 ], _mm256_set1_pd( o_p4[ 1 ][ 3 ] ), r_p4 );
                _mm256_storeu_pd( rb + 1 * r_s, r_p4 );

                r_p4 = _mm256_loadu_pd( rb + 2 * r_s );
                r_p4 = _mm256_fmadd_pd( m_p4[ 0 ], _mm256_set1_pd( o_p4[ 2 ][ 0 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 1 ], _mm256_set1_pd( o_p4[ 2 ][ 1 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 2 ], _mm256_set1_pd( o_p4[ 2 ][ 2 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 3 ], _mm256_set1_pd( o_p4[ 2 ][ 3 ] ), r_p4 );
                _mm256_storeu_pd( rb + 2 * r_s, r_p4 );

                r_p4 = _mm256_loadu_pd( rb + 3 * r_s );
                r_p4 = _mm256_fmadd_pd( m_p4[ 0 ], _mm256_set1_pd( o_p4[ 3 ][ 0 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 1 ], _mm256_set1_pd( o_p4[ 3 ][ 1 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 2 ], _mm256_set1_pd( o_p4[ 3 ][ 2 ] ), r_p4 );
                r_p4 = _mm256_fmadd_pd( m_p4[ 3 ], _mm256_set1_pd( o_p4[ 3 ][ 3 ] ), r_p4 );
                _mm256_storeu_pd( rb + 3 * r_s, r_p4 );
            }
        }
    }
}

static void bmath_mf3_s_mul_avx_microkernel_03( const f3_t* o, sz_t o_s, const f3_t* m, sz_t m_s, f3_t* r, sz_t r_s )
{
    assert( mul_kernel_m_cols == 32 );
    __m256d r_p4[ 8 ];

    __m256d m_p4[ 32 ][ 8 ];

    for( sz_t j = 0; j < 32; j++ )
    {
        const f3_t* mj = m + j * m_s;
        m_p4[ j ][ 0 ] = _mm256_loadu_pd( mj + 0 * 4 );
        m_p4[ j ][ 1 ] = _mm256_loadu_pd( mj + 1 * 4 );
        m_p4[ j ][ 2 ] = _mm256_loadu_pd( mj + 2 * 4 );
        m_p4[ j ][ 3 ] = _mm256_loadu_pd( mj + 3 * 4 );
        m_p4[ j ][ 4 ] = _mm256_loadu_pd( mj + 4 * 4 );
        m_p4[ j ][ 5 ] = _mm256_loadu_pd( mj + 5 * 4 );
        m_p4[ j ][ 6 ] = _mm256_loadu_pd( mj + 6 * 4 );
        m_p4[ j ][ 7 ] = _mm256_loadu_pd( mj + 7 * 4 );
    }

    for( sz_t i = 0; i < mul_kernel_o_rows; i++ )
    {
        const f3_t* oi = o + i * o_s;
              f3_t* ri = r + i * r_s;

        __m256d o_p4 = _mm256_set1_pd( oi[ 0 ] );
        r_p4[ 0 ] = _mm256_mul_pd( m_p4[ 0 ][ 0 ], o_p4 );
        r_p4[ 1 ] = _mm256_mul_pd( m_p4[ 0 ][ 1 ], o_p4 );
        r_p4[ 2 ] = _mm256_mul_pd( m_p4[ 0 ][ 2 ], o_p4 );
        r_p4[ 3 ] = _mm256_mul_pd( m_p4[ 0 ][ 3 ], o_p4 );
        r_p4[ 4 ] = _mm256_mul_pd( m_p4[ 0 ][ 4 ], o_p4 );
        r_p4[ 5 ] = _mm256_mul_pd( m_p4[ 0 ][ 5 ], o_p4 );
        r_p4[ 6 ] = _mm256_mul_pd( m_p4[ 0 ][ 6 ], o_p4 );
        r_p4[ 7 ] = _mm256_mul_pd( m_p4[ 0 ][ 7 ], o_p4 );

        for( sz_t j = 1; j < mul_kernel_o_cols; j++ )
        {
            o_p4 = _mm256_set1_pd( oi[ j ] );
            r_p4[ 0 ] = _mm256_fmadd_pd( m_p4[ j ][ 0 ], o_p4, r_p4[ 0 ] );
            r_p4[ 1 ] = _mm256_fmadd_pd( m_p4[ j ][ 1 ], o_p4, r_p4[ 1 ] );
            r_p4[ 2 ] = _mm256_fmadd_pd( m_p4[ j ][ 2 ], o_p4, r_p4[ 2 ] );
            r_p4[ 3 ] = _mm256_fmadd_pd( m_p4[ j ][ 3 ], o_p4, r_p4[ 3 ] );
            r_p4[ 4 ] = _mm256_fmadd_pd( m_p4[ j ][ 4 ], o_p4, r_p4[ 4 ] );
            r_p4[ 5 ] = _mm256_fmadd_pd( m_p4[ j ][ 5 ], o_p4, r_p4[ 5 ] );
            r_p4[ 6 ] = _mm256_fmadd_pd( m_p4[ j ][ 6 ], o_p4, r_p4[ 6 ] );
            r_p4[ 7 ] = _mm256_fmadd_pd( m_p4[ j ][ 7 ], o_p4, r_p4[ 7 ] );
        }

        _mm256_storeu_pd( ri + 0 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 0 * 4 ), r_p4[ 0 ] ) );
        _mm256_storeu_pd( ri + 1 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 1 * 4 ), r_p4[ 1 ] ) );
        _mm256_storeu_pd( ri + 2 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 2 * 4 ), r_p4[ 2 ] ) );
        _mm256_storeu_pd( ri + 3 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 3 * 4 ), r_p4[ 3 ] ) );
        _mm256_storeu_pd( ri + 4 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 4 * 4 ), r_p4[ 4 ] ) );
        _mm256_storeu_pd( ri + 5 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 5 * 4 ), r_p4[ 5 ] ) );
        _mm256_storeu_pd( ri + 6 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 6 * 4 ), r_p4[ 6 ] ) );
        _mm256_storeu_pd( ri + 7 * 4, _mm256_add_pd( _mm256_loadu_pd( ri + 7 * 4 ), r_p4[ 7 ] ) );
    }
}

static void bmath_mf3_s_f3_t_mul( const f3_t* o, sz_t o_s, sz_t o_r, sz_t o_c, const f3_t* m, sz_t m_s, sz_t m_c, f3_t* r, sz_t r_s )
{
    if( o_r == mul_kernel_o_rows && o_c == mul_kernel_o_cols && m_c == mul_kernel_m_cols )
    {
        #ifdef BMATH_AVX2_FMA
            bmath_mf3_s_mul_avx_microkernel_03( o, o_s, m, m_s, r, r_s );
        #else
            for( sz_t i = 0; i < mul_kernel_o_rows; i++ )
            {
                const f3_t* oi = o + i * o_s;
                      f3_t* ri = r + i * r_s;
                for( sz_t j = 0; j < mul_kernel_o_cols; j++ )
                {
                    const f3_t* mj = m + j * m_s;
                    for( sz_t k = 0; k < mul_kernel_m_cols; k++ )
                    {
                        ri[ k ] += mj[ k ] * oi[ j ];
                    }
                }
            }
        #endif // BMATH_AVX2_FMA

        return;
    }

    if( o_r >= o_c && o_r >= m_c && o_r > mul_kernel_o_rows )
    {
        sz_t mid = ( o_r / ( mul_kernel_o_rows << 1 ) ) * mul_kernel_o_rows;
        if( mid == 0 ) mid++;
        bmath_mf3_s_f3_t_mul( o,             o_s,       mid, o_c, m, m_s, m_c,             r, r_s );
        bmath_mf3_s_f3_t_mul( o + mid * o_s, o_s, o_r - mid, o_c, m, m_s, m_c, r + mid * r_s, r_s );
        return;
    }

    if( o_c >= m_c && o_c > mul_kernel_o_cols )
    {
        sz_t mid = ( o_c / ( mul_kernel_o_cols << 1 ) ) * mul_kernel_o_cols;
        if( mid == 0 ) mid++;
        bmath_mf3_s_f3_t_mul( o,       o_s, o_r,       mid, m,             m_s, m_c, r, r_s );
        bmath_mf3_s_f3_t_mul( o + mid, o_s, o_r, o_c - mid, m + mid * m_s, m_s, m_c, r, r_s );
        return;
    }

    if( m_c > mul_kernel_m_cols )
    {
        sz_t mid = ( m_c / ( mul_kernel_m_cols << 1 ) ) * mul_kernel_m_cols;
        if( mid == 0 ) mid++;
        bmath_mf3_s_f3_t_mul( o, o_s, o_r, o_c, m,       m_s,       mid, r,       r_s );
        bmath_mf3_s_f3_t_mul( o, o_s, o_r, o_c, m + mid, m_s, m_c - mid, r + mid, r_s );
        return;
    }

    /// smaller blocks
    for( sz_t i = 0; i < o_r; i++ )
    {
        const f3_t* oi = o + i * o_s;
              f3_t* ri = r + i * r_s;
        for( sz_t j = 0; j < o_c; j++ )
        {
            f3_t f = oi[ j ];
            const f3_t* mj = m + j * m_s;
            for( sz_t k = 0; k < m_c; k++ )
            {
                ri[ k ] += mj[ k ] * f;
            }
        }
    }
}


void bmath_mf3_s_mul2( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    if( r == o || r == m )
    {
        bmath_mf3_s* buf = bmath_mf3_s_create();
        bmath_mf3_s_set_size( buf, r->rows, r->cols );
        bmath_mf3_s_mul2( o, m, buf );
        bmath_mf3_s_cpy( buf, r );
        bmath_mf3_s_discard( buf );
        return;
    }

    ASSERT( o->cols == m->rows );
    ASSERT( o->rows == r->rows );
    ASSERT( m->cols == r->cols );

    bmath_mf3_s_zro( r );

    bmath_mf3_s_f3_t_mul( o->data, o->stride, o->rows, o->cols, m->data, m->stride, m->cols, r->data, r->stride );
}

//---------------------------------------------------------------------------------------------------------------------
