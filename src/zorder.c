/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "zorder.h"

//----------------------------------------------------------------------------------------------------------------------

#ifdef BMATH_AVX2

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

//----------------------------------------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------------------------------------

static void bmath_mf3_s_zorder_mul_avx_fix_kernel( const f3_t* o, const f3_t* m, f3_t* r )
{
    ASSERT( ( min_rows & 3 ) == 0 );

    #define mul_kernel_m_c4 ( min_rows >> 2 )

    __m256d r_p4[ mul_kernel_m_c4 ];

    for( sz_t i = 0; i < min_rows; i++ )
    {
        const f3_t* oi = o + i * min_rows;
              f3_t* ri = r + i * min_rows;

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ ) r_p4[ k ] = _mm256_set1_pd( 0 );

        for( sz_t j = 0; j < min_rows; j++ )
        {
            __m256d o_p4 = _mm256_set1_pd( oi[ j ] );
            r_p4[ 0 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 0 * 4 ), o_p4, r_p4[ 0 ] );
            r_p4[ 1 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 1 * 4 ), o_p4, r_p4[ 1 ] );
            r_p4[ 2 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 2 * 4 ), o_p4, r_p4[ 2 ] );
            r_p4[ 3 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 3 * 4 ), o_p4, r_p4[ 3 ] );
            r_p4[ 4 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 4 * 4 ), o_p4, r_p4[ 4 ] );
            r_p4[ 5 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 5 * 4 ), o_p4, r_p4[ 5 ] );
            r_p4[ 6 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 6 * 4 ), o_p4, r_p4[ 6 ] );
            r_p4[ 7 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 7 * 4 ), o_p4, r_p4[ 7 ] );
        }

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            _mm256_storeu_pd( ri + k * 4, _mm256_add_pd( _mm256_loadu_pd( ri + k * 4 ), r_p4[ k ] ) );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_zorder_mul_add( const f3_t* o, const f3_t* m, f3_t* r, sz_t rows )
{
    ASSERT( rows >= min_rows );
    if( rows == min_rows )
    {
        bmath_mf3_s_zorder_mul_avx_fix_kernel( o, m, r );
    }
    else
    {
        sz_t hr = rows >> 1;
        sz_t sqr_hr = f3_sqr( hr );
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

//----------------------------------------------------------------------------------------------------------------------

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

//----------------------------------------------------------------------------------------------------------------------

static void bmath_mf3_s_norder_mul_avx_fix_kernel( const f3_t* o, sz_t o_s, const f3_t* m, sz_t m_s, f3_t* r, sz_t r_s  )
{
    ASSERT( ( min_rows & 3 ) == 0 );

    #define mul_kernel_m_c4 ( min_rows >> 2 )

    __m256d m_p4[ min_rows ][ mul_kernel_m_c4 ];
    __m256d r_p4[ mul_kernel_m_c4 ];

    for( sz_t j = 0; j < min_rows; j++ )
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

    for( sz_t i = 0; i < min_rows; i++ )
    {
        const f3_t* oi = o + i * o_s;
              f3_t* ri = r + i * r_s;

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ ) r_p4[ k ] = _mm256_set1_pd( 0 );

        for( sz_t j = 0; j < min_rows; j++ )
        {
            __m256d o_p4 = _mm256_set1_pd( oi[ j ] );
            r_p4[ 0 ] = _mm256_fmadd_pd( m_p4[ j ][ 0 ], o_p4, r_p4[ 0 ] );
            r_p4[ 1 ] = _mm256_fmadd_pd( m_p4[ j ][ 1 ], o_p4, r_p4[ 1 ] );
            r_p4[ 2 ] = _mm256_fmadd_pd( m_p4[ j ][ 2 ], o_p4, r_p4[ 2 ] );
            r_p4[ 3 ] = _mm256_fmadd_pd( m_p4[ j ][ 3 ], o_p4, r_p4[ 3 ] );
            r_p4[ 4 ] = _mm256_fmadd_pd( m_p4[ j ][ 4 ], o_p4, r_p4[ 4 ] );
            r_p4[ 5 ] = _mm256_fmadd_pd( m_p4[ j ][ 5 ], o_p4, r_p4[ 5 ] );
            r_p4[ 6 ] = _mm256_fmadd_pd( m_p4[ j ][ 6 ], o_p4, r_p4[ 6 ] );
            r_p4[ 7 ] = _mm256_fmadd_pd( m_p4[ j ][ 7 ], o_p4, r_p4[ 7 ] );
        }

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            _mm256_storeu_pd( ri + k * 4, _mm256_add_pd( _mm256_loadu_pd( ri + k * 4 ), r_p4[ k ] ) );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_norder_mul_add( const f3_t* o, sz_t o_s, const f3_t* m, sz_t m_s, f3_t* r, sz_t r_s, sz_t rows )
{
    ASSERT( rows >= min_rows );
    if( rows == min_rows )
    {
        bmath_mf3_s_norder_mul_avx_fix_kernel( o, o_s, m, m_s, r, r_s );
    }
    else
    {
        sz_t h = rows >> 1;
        bmath_norder_mul_add( o + o_s * 0 + 0, o_s, m + m_s * 0 + 0, m_s, r + r_s * 0 + 0, r_s, h );
        bmath_norder_mul_add( o + o_s * 0 + 0, o_s, m + m_s * 0 + h, m_s, r + r_s * 0 + h, r_s, h );
        bmath_norder_mul_add( o + o_s * 0 + h, o_s, m + m_s * h + 0, m_s, r + r_s * 0 + 0, r_s, h );
        bmath_norder_mul_add( o + o_s * 0 + h, o_s, m + m_s * h + h, m_s, r + r_s * 0 + h, r_s, h );

        bmath_norder_mul_add( o + o_s * h + 0, o_s, m + m_s * 0 + 0, m_s, r + r_s * h + 0, r_s, h );
        bmath_norder_mul_add( o + o_s * h + 0, o_s, m + m_s * 0 + h, m_s, r + r_s * h + h, r_s, h );
        bmath_norder_mul_add( o + o_s * h + h, o_s, m + m_s * h + 0, m_s, r + r_s * h + 0, r_s, h );
        bmath_norder_mul_add( o + o_s * h + h, o_s, m + m_s * h + h, m_s, r + r_s * h + h, r_s, h );
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_norder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    ASSERT( o->rows == o->cols );
    ASSERT( o->rows == m->rows );
    ASSERT( o->cols == m->cols );
    ASSERT( o->rows == r->rows );
    ASSERT( o->cols == r->cols );
    bmath_mf3_s_zro( r );
    ABS_TIME_TO_STDOUT( bmath_norder_mul_add( o->data, o->stride, m->data, m->stride, r->data, r->stride, o->rows ) );
}

//----------------------------------------------------------------------------------------------------------------------

static void bmath_mf3_s_morder_mul_avx_fix_kernel( const f3_t* o, sz_t o_s, const f3_t* m, f3_t* r, sz_t r_s )
{
    ASSERT( ( min_rows & 3 ) == 0 );

    #define mul_kernel_m_c4 ( min_rows >> 2 )

    __m256d r_p4[ mul_kernel_m_c4 ];

    for( sz_t i = 0; i < min_rows; i++ )
    {
        const f3_t* oi = o + i * o_s;
              f3_t* ri = r + i * r_s;

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ ) r_p4[ k ] = _mm256_set1_pd( 0 );

        for( sz_t j = 0; j < min_rows; j++ )
        {
            __m256d o_p4 = _mm256_set1_pd( oi[ j ] );
            r_p4[ 0 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 0 * 4 ), o_p4, r_p4[ 0 ] );
            r_p4[ 1 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 1 * 4 ), o_p4, r_p4[ 1 ] );
            r_p4[ 2 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 2 * 4 ), o_p4, r_p4[ 2 ] );
            r_p4[ 3 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 3 * 4 ), o_p4, r_p4[ 3 ] );
            r_p4[ 4 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 4 * 4 ), o_p4, r_p4[ 4 ] );
            r_p4[ 5 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 5 * 4 ), o_p4, r_p4[ 5 ] );
            r_p4[ 6 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 6 * 4 ), o_p4, r_p4[ 6 ] );
            r_p4[ 7 ] = _mm256_fmadd_pd( _mm256_loadu_pd( m + j * min_rows + 7 * 4 ), o_p4, r_p4[ 7 ] );
        }

        for( sz_t k = 0; k < mul_kernel_m_c4; k++ )
        {
            _mm256_storeu_pd( ri + k * 4, _mm256_add_pd( _mm256_loadu_pd( ri + k * 4 ), r_p4[ k ] ) );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_morder_mul_add( const f3_t* o, sz_t o_s, const f3_t* m, f3_t* r, sz_t r_s, sz_t rows )
{
    ASSERT( rows >= min_rows );
    if( rows == min_rows )
    {
        bmath_mf3_s_morder_mul_avx_fix_kernel( o, o_s, m, r, r_s );
    }
    else
    {
        sz_t h = rows >> 1;
        sz_t sh = f3_sqr( h );

        bmath_morder_mul_add( o + o_s * 0 + 0, o_s, m + sh * 0, r + r_s * 0 + 0, r_s, h );
        bmath_morder_mul_add( o + o_s * 0 + 0, o_s, m + sh * 1, r + r_s * 0 + h, r_s, h );
        bmath_morder_mul_add( o + o_s * 0 + h, o_s, m + sh * 2, r + r_s * 0 + 0, r_s, h );
        bmath_morder_mul_add( o + o_s * 0 + h, o_s, m + sh * 3, r + r_s * 0 + h, r_s, h );

        bmath_morder_mul_add( o + o_s * h + 0, o_s, m + sh * 0, r + r_s * h + 0, r_s, h );
        bmath_morder_mul_add( o + o_s * h + 0, o_s, m + sh * 1, r + r_s * h + h, r_s, h );
        bmath_morder_mul_add( o + o_s * h + h, o_s, m + sh * 2, r + r_s * h + 0, r_s, h );
        bmath_morder_mul_add( o + o_s * h + h, o_s, m + sh * 3, r + r_s * h + h, r_s, h );

    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_morder_mul( const bmath_mf3_s* o, const bmath_mf3_s* m, bmath_mf3_s* r )
{
    ASSERT( o->rows == o->cols );
    ASSERT( o->rows == m->rows );
    ASSERT( o->cols == m->cols );
    ASSERT( o->rows == r->rows );
    ASSERT( o->cols == r->cols );

    f3_t* mz = bcore_u_alloc( sizeof( f3_t ), NULL, m->rows * m->cols, NULL );
    bmath_zorder_fwd_cpy( m->data, m->stride, m->rows, mz );
    bmath_mf3_s_zro( r );

    ABS_TIME_TO_STDOUT( bmath_morder_mul_add( o->data, o->stride, mz, r->data, r->stride, o->rows ) );

    bcore_u_alloc( sizeof( f3_t ), mz, 0, NULL );

}

//----------------------------------------------------------------------------------------------------------------------

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

#endif // BMATH_AVX2

//----------------------------------------------------------------------------------------------------------------------

