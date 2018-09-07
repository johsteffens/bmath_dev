/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "zorder.h"

//---------------------------------------------------------------------------------------------------------------------

#define min_rows 32

void bmath_zorder_fwd_cpy( const f3_t* o, sz_t o_stride, sz_t o_rows, f3_t* r )
{
    ASSERT( o_rows >= min_rows );
    if( o_rows == min_rows )
    {
        for( sz_t i = 0; i < min_rows; i++ )
        {
            const f3_t* oi = o + i * o_stride;
                  f3_t* ri = r + i * min_rows;
            bcore_u_memcpy( sizeof( f3_t ), ri, oi, min_rows );
        }
    }
    else
    {
        sz_t hr = o_rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
        bmath_zorder_fwd_cpy( o,                         o_stride, hr, r + sqr_hr * 0 );
        bmath_zorder_fwd_cpy( o + hr,                    o_stride, hr, r + sqr_hr * 1 );
        bmath_zorder_fwd_cpy( o + hr * o_stride,         o_stride, hr, r + sqr_hr * 2 );
        bmath_zorder_fwd_cpy( o + hr * ( o_stride + 1 ), o_stride, hr, r + sqr_hr * 3 );
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_zorder_rev_cpy( f3_t* o, sz_t o_stride, sz_t o_rows, const f3_t* r )
{
    ASSERT( o_rows >= min_rows );
    if( o_rows == min_rows )
    {
        for( sz_t i = 0; i < o_rows; i++ )
        {
                  f3_t* oi = o + i * o_stride;
            const f3_t* ri = r + i * o_rows;
            bcore_u_memcpy( sizeof( f3_t ), oi, ri, o_rows );
        }
    }
    else
    {
        sz_t hr = o_rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
        bmath_zorder_rev_cpy( o,                         o_stride, hr, r + sqr_hr * 0 );
        bmath_zorder_rev_cpy( o + hr,                    o_stride, hr, r + sqr_hr * 1 );
        bmath_zorder_rev_cpy( o + hr * o_stride,         o_stride, hr, r + sqr_hr * 2 );
        bmath_zorder_rev_cpy( o + hr * ( o_stride + 1 ), o_stride, hr, r + sqr_hr * 3 );
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_zorder_fwd_cpy2( const f3_t* o, sz_t o_stride, sz_t o_rows, f3_t* r )
{
    ASSERT( o_rows >= min_rows );
    if( o_rows == min_rows )
    {
        for( sz_t i = 0; i < min_rows; i++ )
        {
            const f3_t* oi = o + i * o_stride;
                  f3_t* ri = r + i * min_rows;
            bcore_u_memcpy( sizeof( f3_t ), ri, oi, min_rows );
        }
    }
    else
    {
        sz_t hr = o_rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
        bmath_zorder_fwd_cpy2( o,                         o_stride, hr, r + sqr_hr * 0 );
        bmath_zorder_fwd_cpy2( o + hr * o_stride,         o_stride, hr, r + sqr_hr * 1 );
        bmath_zorder_fwd_cpy2( o + hr,                    o_stride, hr, r + sqr_hr * 2 );
        bmath_zorder_fwd_cpy2( o + hr * ( o_stride + 1 ), o_stride, hr, r + sqr_hr * 3 );
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_zorder_rev_cpy2( f3_t* o, sz_t o_stride, sz_t o_rows, const f3_t* r )
{
    ASSERT( o_rows >= min_rows );
    if( o_rows == min_rows )
    {
        for( sz_t i = 0; i < o_rows; i++ )
        {
                  f3_t* oi = o + i * o_stride;
            const f3_t* ri = r + i * o_rows;
            bcore_u_memcpy( sizeof( f3_t ), oi, ri, o_rows );
        }
    }
    else
    {
        sz_t hr = o_rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
        bmath_zorder_rev_cpy( o,                         o_stride, hr, r + sqr_hr * 0 );
        bmath_zorder_rev_cpy( o + hr * o_stride,         o_stride, hr, r + sqr_hr * 1 );
        bmath_zorder_rev_cpy( o + hr,                    o_stride, hr, r + sqr_hr * 2 );
        bmath_zorder_rev_cpy( o + hr * ( o_stride + 1 ), o_stride, hr, r + sqr_hr * 3 );
    }
}

//---------------------------------------------------------------------------------------------------------------------

static void bmath_mf3_s_mul_avx_fix_kernel( const f3_t* o, const f3_t* m, f3_t* r )
{
    ASSERT( ( min_rows & 3 ) == 0 );

    #define mul_kernel_m_c4 ( min_rows >> 2 )

    __m256d r_p4[ mul_kernel_m_c4   ];
    __m256d m_p4[ min_rows ][ mul_kernel_m_c4 ];

    for( sz_t j = 0; j < min_rows; j++ )
    {
        const f3_t* mj = m + j * min_rows;
        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            m_p4[ j ][ k ] = _mm256_loadu_pd( mj + k * 4 );
        }
    }

    for( sz_t i = 0; i < min_rows; i++ )
    {
        const f3_t* oi = o + i * min_rows;
              f3_t* ri = r + i * min_rows;

        __m256d o_p4 = _mm256_set1_pd( oi[ 0 ] );

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            r_p4[ k ] = _mm256_mul_pd( m_p4[ 0 ][ k ], o_p4 );
        }

        for( sz_t j = 1; j < min_rows; j++ )
        {
            o_p4 = _mm256_set1_pd( oi[ j ] );
            for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
            {
                #ifdef BMATH_AVX2_FMA
                    r_p4[ k ] = _mm256_fmadd_pd( m_p4[ j ][ k ], o_p4, r_p4[ k ] );
                #else
                    r_p4[ k ] = _mm256_add_pd( _mm256_mul_pd( m_p4[ j ][ k ], o_p4 ), r_p4[ k ] );
                #endif // BMATH_AVX2_FMA
            }
        }

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            _mm256_storeu_pd( ri + k * 4, _mm256_add_pd( _mm256_loadu_pd( ri + k * 4 ), r_p4[ k ] ) );
        }
    }
}

void bmath_zorder_mul_add( const f3_t* o, const f3_t* m, f3_t* r, sz_t rows )
{
    ASSERT( rows >= min_rows );
    if( rows == min_rows )
    {
        bmath_mf3_s_mul_avx_fix_kernel( o, m, r );
        /*
        for( sz_t i = 0; i < min_rows; i++ )
        {
            const f3_t* oi = o + i * min_rows;
                  f3_t* ri = r + i * min_rows;
            for( sz_t j = 0; j < min_rows; j++ )
            {
                const f3_t* mj = m + j * min_rows;
                for( sz_t k = 0; k < min_rows; k++ ) ri[ k ] += mj[ k ] * oi[ j ];
            }
        }
        */
    }
    else
    {
        sz_t hr = rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
//        bcore_msg_fa( "#<sz_t>\n", hr );

        bmath_zorder_mul_add( o + sqr_hr * 0, m + sqr_hr * 0, r + sqr_hr * 0, hr );
        bmath_zorder_mul_add( o + sqr_hr * 0, m + sqr_hr * 1, r + sqr_hr * 1, hr );
        bmath_zorder_mul_add( o + sqr_hr * 1, m + sqr_hr * 2, r + sqr_hr * 0, hr );
        bmath_zorder_mul_add( o + sqr_hr * 1, m + sqr_hr * 3, r + sqr_hr * 1, hr );

        bmath_zorder_mul_add( o + sqr_hr * 2, m + sqr_hr * 0, r + sqr_hr * 2, hr );
        bmath_zorder_mul_add( o + sqr_hr * 2, m + sqr_hr * 1, r + sqr_hr * 3, hr );
        bmath_zorder_mul_add( o + sqr_hr * 3, m + sqr_hr * 2, r + sqr_hr * 2, hr );
        bmath_zorder_mul_add( o + sqr_hr * 3, m + sqr_hr * 3, r + sqr_hr * 3, hr );
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_zorder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    ASSERT( o->rows == o->cols );
    ASSERT( o->rows == m->rows );
    ASSERT( o->cols == m->cols );
    ASSERT( o->rows == r->rows );
    ASSERT( o->cols == r->cols );

    f3_t* oz = bcore_u_alloc( sizeof( f3_t ), NULL, o->rows * o->cols, NULL );
    f3_t* mz = bcore_u_alloc( sizeof( f3_t ), NULL, m->rows * m->cols, NULL );
    f3_t* rz = bcore_u_alloc( sizeof( f3_t ), NULL, r->rows * r->cols, NULL );

    bmath_zorder_fwd_cpy( o->data, o->stride, o->rows, oz );
    bmath_zorder_fwd_cpy( m->data, m->stride, m->rows, mz );

    bcore_u_memzero( sizeof( f3_t ), rz, r->rows * r->cols );

    ABS_TIME_TO_STDOUT( bmath_zorder_mul_add( oz, mz, rz, o->rows ) );

    bmath_zorder_rev_cpy( r->data, r->stride, r->rows, rz );

    bcore_u_alloc( sizeof( f3_t ), oz, 0, NULL );
    bcore_u_alloc( sizeof( f3_t ), mz, 0, NULL );
    bcore_u_alloc( sizeof( f3_t ), rz, 0, NULL );

}

//---------------------------------------------------------------------------------------------------------------------

void bmath_zorder_test( void )
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m2 );
    BCORE_LIFE_CREATE( bmath_mf3_s, m3 );

    sz_t n = 16;
    bmath_mf3_s_set_size( m1, n, n );
    bmath_mf3_s_set_size( m2, n, n );
    bmath_mf3_s_set_size( m3, n, n );

    for( sz_t i = 0; i < n; i++ )
    {
        for( sz_t j = 0; j < n; j++ )
        {
            m1->data[ i * m1->stride + j ] = i * m1->stride + j;
        }
    }

    bmath_mf3_s_to_stdout( m1 );
    bmath_zorder_fwd_cpy( m1->data, m1->stride, m1->rows, m2->data );
    bmath_mf3_s_to_stdout( m2 );
    bmath_zorder_rev_cpy( m3->data, m3->stride, m3->rows, m2->data );
    bmath_mf3_s_to_stdout( m3 );

    BCORE_LIFE_DOWN();
}

//---------------------------------------------------------------------------------------------------------------------

