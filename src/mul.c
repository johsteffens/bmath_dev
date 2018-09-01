/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "mul.h"

//---------------------------------------------------------------------------------------------------------------------

static void bmath_f3_t_mat_mul_add( const f3_t* o, sz_t o_stride, sz_t o_rows, sz_t o_cols, const f3_t* b, sz_t b_stride, sz_t b_cols, f3_t* r, sz_t r_stride )
{
    const sz_t block_size = 10 * 4; // must be multiple of 4

    if( o_rows >= b_cols && o_rows > block_size )
    {
        sz_t mid = o_rows >> 1;
        bmath_f3_t_mat_mul_add( o,                  o_stride, mid,          o_cols, b, b_stride, b_cols, r,                  r_stride );
        bmath_f3_t_mat_mul_add( o + mid * o_stride, o_stride, o_rows - mid, o_cols, b, b_stride, b_cols, r + mid * r_stride, r_stride );
    }
    else if( b_cols > block_size )
    {
        sz_t mid = b_cols >> 1;
        bmath_f3_t_mat_mul_add( o, o_stride, o_rows, o_cols, b,       b_stride, mid,          r,       r_stride );
        bmath_f3_t_mat_mul_add( o, o_stride, o_rows, o_cols, b + mid, b_stride, b_cols - mid, r + mid, r_stride );
    }
    else
    {
        sz_t l = 0;
        const sz_t block_size_p4 = block_size >> 2;

        #ifdef BMATH_AVX2
            for( ; l <= o_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < b_cols; i++ )
                {
                    const f3_t* vb = b + i + b_stride * l;
                          f3_t* vr = r + i;

                    __m256d b_p4[ block_size_p4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        b_p4[ k ][ 0 ] = vb[ ( k * 4 + 0 ) * b_stride ];
                        b_p4[ k ][ 1 ] = vb[ ( k * 4 + 1 ) * b_stride ];
                        b_p4[ k ][ 2 ] = vb[ ( k * 4 + 2 ) * b_stride ];
                        b_p4[ k ][ 3 ] = vb[ ( k * 4 + 3 ) * b_stride ];
                    }

                    for( sz_t j = 0; j < o_rows; j++ )
                    {
                        const f3_t* vo = o + j * o_stride + l;

                        __m256d r_p4 = _mm256_mul_pd( b_p4[ 0 ], _mm256_loadu_pd( vo ) );
                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            #ifdef BMATH_AVX2_FMA
                                r_p4 = _mm256_fmadd_pd( b_p4[ k ], _mm256_loadu_pd( vo + k * 4 ), r_p4 );
                            #else
                                r_p4 = _mm256_add_pd( _mm256_mul_pd( b_p4[ k ], _mm256_loadu_pd( vo + k * 4 ) ), r_p4 );
                            #endif // BMATH_AVX2_FMA
                        }
                        r_p4 = _mm256_hadd_pd( r_p4, r_p4 );

                        vr[ j * r_stride ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #else  // fallback code
            for( ; l <= o_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < b_cols; i++ )
                {
                    const f3_t* vb = b + i + b_stride * l;
                          f3_t* vr = r + i;

                    f3_t b_p4[ block_size_p4 ][ 4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        b_p4[ k ][ 0 ] = vb[ ( k * 4 + 0 ) * b_stride ];
                        b_p4[ k ][ 1 ] = vb[ ( k * 4 + 1 ) * b_stride ];
                        b_p4[ k ][ 2 ] = vb[ ( k * 4 + 2 ) * b_stride ];
                        b_p4[ k ][ 3 ] = vb[ ( k * 4 + 3 ) * b_stride ];
                    }

                    for( sz_t j = 0; j < o_rows; j++ )
                    {
                        const f3_t* vo = o + j * o_stride + l;

                        f3_t r_p4[ 4 ];
                        r_p4[ 0 ] = b_p4[ 0 ][ 0 ] * vo[ 0 ];
                        r_p4[ 1 ] = b_p4[ 0 ][ 1 ] * vo[ 1 ];
                        r_p4[ 2 ] = b_p4[ 0 ][ 2 ] * vo[ 2 ];
                        r_p4[ 3 ] = b_p4[ 0 ][ 3 ] * vo[ 3 ];

                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            r_p4[ 0 ] += b_p4[ k ][ 0 ] * vo[ k * 4 + 0 ];
                            r_p4[ 1 ] += b_p4[ k ][ 1 ] * vo[ k * 4 + 1 ];
                            r_p4[ 2 ] += b_p4[ k ][ 2 ] * vo[ k * 4 + 2 ];
                            r_p4[ 3 ] += b_p4[ k ][ 3 ] * vo[ k * 4 + 3 ];
                        }

                        r_p4[ 0 ] += r_p4[ 1 ];
                        r_p4[ 2 ] += r_p4[ 3 ];

                        vr[ j * r_stride ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #endif // BMATH_AVX2

        for( sz_t i = 0; i < o_rows; i++ )
        {
            const f3_t* vo = o + o_stride * i;
                  f3_t* vr = r + i * r_stride;
            for( sz_t j = 0; j < b_cols; j++ )
            {
                const f3_t* vb = b + j;
                f3_t sum[ 4 ] = { 0, 0, 0, 0 };
                sz_t k = l;
                for( ; k <= o_cols - 4; k += 4 )
                {
                    sum[ 0 ] += vo[ k + 0 ] * vb[ ( k + 0 ) * b_stride ];
                    sum[ 1 ] += vo[ k + 1 ] * vb[ ( k + 1 ) * b_stride ];
                    sum[ 2 ] += vo[ k + 2 ] * vb[ ( k + 2 ) * b_stride ];
                    sum[ 3 ] += vo[ k + 3 ] * vb[ ( k + 3 ) * b_stride ];
                }
                for( ; k < o_cols; k++ )
                {
                    sum[ 0 ] += vo[ k ] * vb[ k * b_stride ];
                }

                vr[ j ] += sum[ 0 ] + sum[ 1 ] + sum[ 2 ] + sum[ 3 ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul2( const bmath_mf3_s* o, const bmath_mf3_s* b, bmath_mf3_s* r )
{
    if( r == o || r == b )
    {
        bmath_mf3_s* buf = bmath_mf3_s_create();
        bmath_mf3_s_set_size( buf, r->rows, r->cols );
        bmath_mf3_s_mul2( o, b, buf );
        bmath_mf3_s_cpy( buf, r );
        bmath_mf3_s_discard( buf );
        return;
    }

    ASSERT( o->cols == b->rows );
    ASSERT( o->rows == r->rows );
    ASSERT( b->cols == r->cols );

    bmath_mf3_s_zro( r );

    bmath_f3_t_mat_mul_add( o->data, o->stride, o->rows, o->cols, b->data, b->stride, b->cols, r->data, r->stride );
}

//---------------------------------------------------------------------------------------------------------------------

static void bmath_f3_t_mat_mul_htp_add( const f3_t* o, sz_t o_stride, sz_t o_rows, sz_t o_cols, const f3_t* b, sz_t b_stride, sz_t b_rows, f3_t* r, sz_t r_stride, bl_t sym )
{
    const sz_t block_size = 8 * 4; // must be multiple of 4

    if( o_rows >= b_rows && o_rows > block_size )
    {
        sz_t mid = o_rows >> 1;
        bmath_f3_t_mat_mul_htp_add( o,                  o_stride, mid,          o_cols, b, b_stride, b_rows, r,                  r_stride, sym );
        bmath_f3_t_mat_mul_htp_add( o + mid * o_stride, o_stride, o_rows - mid, o_cols, b, b_stride, b_rows, r + mid * r_stride, r_stride, sym );
    }
    else if( b_rows > block_size )
    {
        sz_t mid = b_rows >> 1;
        bmath_f3_t_mat_mul_htp_add( o, o_stride, o_rows, o_cols, b, b_stride, mid, r, r_stride, sym );

        if( !sym || o != b ) // in case of symmetry skip upper triangle of r
        {
            bmath_f3_t_mat_mul_htp_add( o, o_stride, o_rows, o_cols, b + mid * b_stride, b_stride, b_rows - mid, r + mid, r_stride, sym );
        }
    }
    else
    {
        sz_t l = 0;
        const sz_t block_size_p4 = block_size >> 2;

        #ifdef BMATH_AVX2
            for( ; l <= o_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < o_rows; i++ )
                {
                    const f3_t* vo = o + o_stride * i + l;
                          f3_t* vr = r + r_stride * i;

                    __m256d o_p4[ block_size_p4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        o_p4[ k ] = _mm256_loadu_pd( vo + k * 4 );
                    }

                    for( sz_t j = 0; j < b_rows; j++ )
                    {
                        const f3_t* vb = b + j * b_stride + l;

                        __m256d r_p4 = _mm256_mul_pd( o_p4[ 0 ], _mm256_loadu_pd( vb ) );
                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            #ifdef BMATH_AVX2_FMA
                                r_p4 = _mm256_fmadd_pd( o_p4[ k ], _mm256_loadu_pd( vb + k * 4 ), r_p4 );
                            #else
                                r_p4 = _mm256_add_pd( _mm256_mul_pd( o_p4[ k ], _mm256_loadu_pd( vb + k * 4 ) ), r_p4 );
                            #endif // BMATH_AVX2_FMA
                        }
                        r_p4 = _mm256_hadd_pd( r_p4, r_p4 );

                        vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #else  // fallback code
            for( ; l <= o_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < o_rows; i++ )
                {
                    const f3_t* vo = o + o_stride * i + l;
                          f3_t* vr = r + r_stride * i;

                    f3_t o_p4[ block_size_p4 ][ 4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        o_p4[ k ][ 0 ] = vo[ k * 4 + 0 ];
                        o_p4[ k ][ 1 ] = vo[ k * 4 + 1 ];
                        o_p4[ k ][ 2 ] = vo[ k * 4 + 2 ];
                        o_p4[ k ][ 3 ] = vo[ k * 4 + 3 ];
                    }

                    for( sz_t j = 0; j < b_rows; j++ )
                    {
                        const f3_t* vb = b + j * b_stride + l;

                        f3_t r_p4[ 4 ];
                        r_p4[ 0 ] = o_p4[ 0 ][ 0 ] * vb[ 0 ];
                        r_p4[ 1 ] = o_p4[ 0 ][ 1 ] * vb[ 1 ];
                        r_p4[ 2 ] = o_p4[ 0 ][ 2 ] * vb[ 2 ];
                        r_p4[ 3 ] = o_p4[ 0 ][ 3 ] * vb[ 3 ];

                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            r_p4[ 0 ] += o_p4[ k ][ 0 ] * vb[ k * 4 + 0 ];
                            r_p4[ 1 ] += o_p4[ k ][ 1 ] * vb[ k * 4 + 1 ];
                            r_p4[ 2 ] += o_p4[ k ][ 2 ] * vb[ k * 4 + 2 ];
                            r_p4[ 3 ] += o_p4[ k ][ 3 ] * vb[ k * 4 + 3 ];
                        }

                        r_p4[ 0 ] += r_p4[ 1 ];
                        r_p4[ 2 ] += r_p4[ 3 ];

                        vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #endif // BMATH_AVX2

        for( sz_t i = 0; i < o_rows; i++ )
        {
            const f3_t* vo = o + o_stride * i;
                  f3_t* vr = r + r_stride * i;
            for( sz_t j = 0; j < b_rows; j++ )
            {
                const f3_t* vb = b + b_stride * j;
                f3_t sum[ 4 ] = { 0, 0, 0, 0 };
                sz_t k = l;
                for( ; k <= o_cols - 4; k += 4 )
                {
                    sum[ 0 ] += vo[ k + 0 ] * vb[ k + 0 ];
                    sum[ 1 ] += vo[ k + 1 ] * vb[ k + 1 ];
                    sum[ 2 ] += vo[ k + 2 ] * vb[ k + 2 ];
                    sum[ 3 ] += vo[ k + 3 ] * vb[ k + 3 ];
                }
                for( ; k < o_cols; k++ )
                {
                    sum[ 0 ] += vo[ k ] * vb[ k ];
                }

                vr[ j ] += sum[ 0 ] + sum[ 1 ] + sum[ 2 ] + sum[ 3 ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_htp2( const bmath_mf3_s* o, const bmath_mf3_s* b, bmath_mf3_s* r )
{
    if( r == o || r == b )
    {
        bmath_mf3_s* buf = bmath_mf3_s_create();
        bmath_mf3_s_set_size( buf, r->rows, r->cols );
        bmath_mf3_s_mul_htp2( o, b, buf );
        bmath_mf3_s_cpy( buf, r );
        bmath_mf3_s_discard( buf );
        return;
    }

    ASSERT( o->cols == b->cols );
    ASSERT( o->rows == r->rows );
    ASSERT( b->rows == r->cols );

    bmath_mf3_s_zro( r );

    bl_t symmetry = ( o == b );

    bmath_f3_t_mat_mul_htp_add( o->data, o->stride, o->rows, o->cols, b->data, b->stride, b->rows, r->data, r->stride, symmetry );

    if( symmetry )
    {
        for( sz_t i = 0; i < r->rows; i++ )
        {
            for( sz_t j = 0; j < i; j++ )
            {
                r->data[ j * r->stride + i ] = r->data[ i * r->stride + j ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------



