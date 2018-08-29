/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "mul.h"

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_add( const f3_t* o1, sz_t o1_stride, sz_t o1_rows, sz_t o1_cols, const f3_t* o2, sz_t o2_stride, sz_t o2_cols, f3_t* r, sz_t r_stride )
{
    const sz_t block_size = 32;
    if( o1_rows >= o1_cols && o1_rows >= o2_cols && o1_rows > block_size )
    {
        sz_t mid = o1_rows >> 1;
        bmath_mf3_s_mul_add( o1,                   o1_stride, mid,           o1_cols, o2, o2_stride, o2_cols, r,                  r_stride );
        bmath_mf3_s_mul_add( o1 + mid * o1_stride, o1_stride, o1_rows - mid, o1_cols, o2, o2_stride, o2_cols, r + mid * r_stride, r_stride );
    }
    else if( o1_cols >= o2_cols &&  o1_cols > block_size )
    {
        sz_t mid = o1_cols >> 1;
        bmath_mf3_s_mul_add( o1,       o1_stride, o1_rows, mid,           o2,                   o2_stride, o2_cols, r, r_stride );
        bmath_mf3_s_mul_add( o1 + mid, o1_stride, o1_rows, o1_cols - mid, o2 + mid * o2_stride, o2_stride, o2_cols, r, r_stride );
    }
    else if( o2_cols > block_size )
    {
        sz_t mid = o2_cols >> 1;
        bmath_mf3_s_mul_add( o1, o1_stride, o1_rows, o1_cols, o2,       o2_stride, mid,           r,       r_stride );
        bmath_mf3_s_mul_add( o1, o1_stride, o1_rows, o1_cols, o2 + mid, o2_stride, o2_cols - mid, r + mid, r_stride );
    }
    else
    {
        for( sz_t i = 0; i < o1_rows; i++ )
        {
            const f3_t* vo1 = o1 + o1_stride * i;
                  f3_t* vr  = r  +  r_stride * i;
            for( sz_t j = 0; j < o2_cols; j++ )
            {
                const f3_t* vo2 = o2 + j;
                for( sz_t k = 0; k < o1_cols; k++ )
                {
                    vr[ j ] += vo1[ k ] * vo2[ k * o2_stride ];
                }
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

/// 16x16 blocks
void bmath_mf3_s_mul_16( const f3_t* o1, sz_t o1_stride, const f3_t* o2, sz_t o2_stride, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i < 16; i++ )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;
        for( sz_t j = 0; j < 16; j++ )
        {
            const f3_t* vo2 = o2 + j;
            for( sz_t k = 0; k < 16; k++ )
            {
                vr[ j ] += vo1[ k ] * vo2[ k * o2_stride ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_add2( const f3_t* o1, sz_t o1_stride, sz_t o1_rows, sz_t o1_cols, const f3_t* o2, sz_t o2_stride, sz_t o2_cols, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i <= o1_rows - 16; i += 16 )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;
        for( sz_t j = 0; j <= o2_cols - 16; j += 16 )
        {
            const f3_t* vo2 = o2 + j;
            for( sz_t k = 0; k <= o1_cols - 16; k += 16 )
            {
                bmath_mf3_s_mul_16( vo1 + k, o1_stride, vo2 + k * o2_stride, o2_stride, vr + j, r_stride );
//                bmath_mf3_s_mul_16x( vo1 + k, o1_stride, vo2 + k * o2_stride, o2_stride, vr + j, r_stride );
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul2( const bmath_mf3_s* o, const bmath_mf3_s* op, bmath_mf3_s* res )
{
    if( res == o || res == op )
    {
        bmath_mf3_s* buf = bmath_mf3_s_create();
        bmath_mf3_s_set_size( buf, res->rows, res->cols );
        bmath_mf3_s_mul2( o, op, buf );
        bmath_mf3_s_cpy( buf, res );
        bmath_mf3_s_discard( buf );
        return;
    }

    ASSERT(  o->cols ==  op->rows );
    ASSERT(  o->rows == res->rows );
    ASSERT( op->cols == res->cols );

    bmath_mf3_s_zro( res );

    bmath_mf3_s_mul_add( o->data, o->stride, o->rows, o->cols, op->data, op->stride, op->cols, res->data, res->stride );
//    bmath_mf3_s_mul_add2( o->data, o->stride, o->rows, o->cols, op->data, op->stride, op->cols, res->data, res->stride );

}

//---------------------------------------------------------------------------------------------------------------------

/// 16x16 blocks
void bmath_mf3_s_mul_htp_16( const f3_t* o1, sz_t o1_stride, const f3_t* o2, sz_t o2_stride, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i < 16; i++ )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;
        for( sz_t j = 0; j < 16; j++ )
        {
            const f3_t* vo2 = o2 + j * o2_stride;
            for( sz_t k = 0; k < 16; k++ )
            {
                vr[ j ] += vo1[ k ] * vo2[ k ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

/// 16x16 blocks
void bmath_mf3_s_mul_htp_16x( const f3_t* o1, sz_t o1_stride, const f3_t* o2, sz_t o2_stride, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i < 16; i++ )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;

        __m256d o1_p4[ 4 ];
        o1_p4[ 0 ] = _mm256_loadu_pd( vo1 + 0 * 4 );
        o1_p4[ 1 ] = _mm256_loadu_pd( vo1 + 1 * 4 );
        o1_p4[ 2 ] = _mm256_loadu_pd( vo1 + 2 * 4 );
        o1_p4[ 3 ] = _mm256_loadu_pd( vo1 + 3 * 4 );

        for( sz_t j = 0; j < 16; j += 2 )
        {
            const f3_t* vo2 = o2 + j * o2_stride;

            __m256d r0_p4 = { 0, 0, 0, 0 };
            __m256d r1_p4 = { 0, 0, 0, 0 };
            r0_p4 = _mm256_fmadd_pd( o1_p4[ 0 ], _mm256_loadu_pd( vo2 +             0 * 4 ), r0_p4 );
            r1_p4 = _mm256_fmadd_pd( o1_p4[ 0 ], _mm256_loadu_pd( vo2 + o2_stride + 0 * 4 ), r1_p4 );
            r0_p4 = _mm256_fmadd_pd( o1_p4[ 1 ], _mm256_loadu_pd( vo2 +             1 * 4 ), r0_p4 );
            r1_p4 = _mm256_fmadd_pd( o1_p4[ 1 ], _mm256_loadu_pd( vo2 + o2_stride + 1 * 4 ), r1_p4 );
            r0_p4 = _mm256_fmadd_pd( o1_p4[ 2 ], _mm256_loadu_pd( vo2 +             2 * 4 ), r0_p4 );
            r1_p4 = _mm256_fmadd_pd( o1_p4[ 2 ], _mm256_loadu_pd( vo2 + o2_stride + 2 * 4 ), r1_p4 );
            r0_p4 = _mm256_fmadd_pd( o1_p4[ 3 ], _mm256_loadu_pd( vo2 +             3 * 4 ), r0_p4 );
            r1_p4 = _mm256_fmadd_pd( o1_p4[ 3 ], _mm256_loadu_pd( vo2 + o2_stride + 3 * 4 ), r1_p4 );


            r0_p4 = _mm256_hadd_pd( r0_p4, r1_p4 );

            vr[ j     ] += r0_p4[ 0 ] + r0_p4[ 2 ];
            vr[ j + 1 ] += r0_p4[ 1 ] + r0_p4[ 3 ];
        }
    }
}

/// 32x32 blocks
void bmath_mf3_s_mul_htp_32x( const f3_t* o1, sz_t o1_stride, const f3_t* o2, sz_t o2_stride, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i < 32; i++ )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;

        __m256d o1_p4[ 8 ];

        for( sz_t k = 0; k < 8; k++ ) o1_p4[ k ] = _mm256_loadu_pd( vo1 + k * 4 );

        for( sz_t j = 0; j < 32; j++ )
        {
            const f3_t* vo2 = o2 + j * o2_stride;

            __m256d r_p4 = _mm256_mul_pd( o1_p4[ 0 ], _mm256_loadu_pd( vo2 ) );
            for( sz_t k = 1; k < 8; k++ ) r_p4 = _mm256_fmadd_pd( o1_p4[ k ], _mm256_loadu_pd( vo2 + k * 4 ), r_p4 );
            r_p4 = _mm256_hadd_pd( r_p4, r_p4 );

            vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

/// 64x64 blocks
void bmath_mf3_s_mul_htp_64x( const f3_t* o1, sz_t o1_stride, const f3_t* o2, sz_t o2_stride, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i < 64; i++ )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;

        __m256d o1_p4[ 16 ];

        for( sz_t k = 0; k < 16; k++ ) o1_p4[ k ] = _mm256_loadu_pd( vo1 + k * 4 );

        for( sz_t j = 0; j < 64; j++ )
        {
            const f3_t* vo2 = o2 + j * o2_stride;

            __m256d r_p4 = { 0, 0, 0, 0 };

            for( sz_t k = 0; k < 16; k++ ) r_p4 = _mm256_fmadd_pd( o1_p4[ k ], _mm256_loadu_pd( vo2 + k * 4 ), r_p4 );
            r_p4 = _mm256_hadd_pd( r_p4, r_p4 );

            vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_add_htp2( const f3_t* o1, sz_t o1_stride, sz_t o1_rows, sz_t o1_cols, const f3_t* o2, sz_t o2_stride, sz_t o2_rows, f3_t* r, sz_t r_stride )
{
    for( sz_t i = 0; i <= o1_rows - 32; i += 32 )
    {
        const f3_t* vo1 = o1 + o1_stride * i;
              f3_t* vr  = r  +  r_stride * i;
        for( sz_t j = 0; j <= o2_rows - 32; j += 32 )
        {
            const f3_t* vo2 = o2 + j * o2_stride;
            for( sz_t k = 0; k <= o1_cols - 32; k += 32 )
            {
//                bmath_mf3_s_mul_htp_32( vo1 + k, o1_stride, vo2 + k, o2_stride, vr + j, r_stride );
                bmath_mf3_s_mul_htp_32x( vo1 + k, o1_stride, vo2 + k, o2_stride, vr + j, r_stride );
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_add_htp( const f3_t* o1, sz_t o1_stride, sz_t o1_rows, sz_t o1_cols, const f3_t* o2, sz_t o2_stride, sz_t o2_rows, f3_t* r, sz_t r_stride )
{
    const sz_t block_size = 8 * 4; // must be multiple of 4

    if( o1_rows > block_size )
    {
        sz_t mid = o1_rows >> 1;
        bmath_mf3_s_mul_add_htp( o1,                   o1_stride, mid,           o1_cols, o2, o2_stride, o2_rows, r,                  r_stride );
        bmath_mf3_s_mul_add_htp( o1 + mid * o1_stride, o1_stride, o1_rows - mid, o1_cols, o2, o2_stride, o2_rows, r + mid * r_stride, r_stride );
    }
    else if( o2_rows > block_size )
    {
        sz_t mid = o2_rows >> 1;
        bmath_mf3_s_mul_add_htp( o1, o1_stride, o1_rows, o1_cols, o2,                   o2_stride, mid,           r,       r_stride );
        bmath_mf3_s_mul_add_htp( o1, o1_stride, o1_rows, o1_cols, o2 + mid * o2_stride, o2_stride, o2_rows - mid, r + mid, r_stride );
    }
    else
    {
        sz_t l = 0;
        const sz_t block_size_p4 = block_size >> 2;

        #ifdef BMATH_AVX2
            for( ; l <= o1_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < o1_rows; i++ )
                {
                    const f3_t* vo1 = o1 + o1_stride * i + l;
                          f3_t* vr  = r  +  r_stride * i;

                    __m256d o1_p4[ block_size_p4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        o1_p4[ k ] = _mm256_loadu_pd( vo1 + k * 4 );
                    }

                    for( sz_t j = 0; j < o2_rows; j++ )
                    {
                        const f3_t* vo2 = o2 + j * o2_stride + l;

                        __m256d r_p4 = _mm256_mul_pd( o1_p4[ 0 ], _mm256_loadu_pd( vo2 ) );
                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            #ifdef BMATH_AVX2_FMA
                                r_p4 = _mm256_fmadd_pd( o1_p4[ k ], _mm256_loadu_pd( vo2 + k * 4 ), r_p4 );
                            #else
                                r_p4 = _mm256_add_pd( _mm256_mul_pd( o1_p4[ k ], _mm256_loadu_pd( vo2 + k * 4 ) ), r_p4 );
                            #endif // BMATH_AVX2_FMA
                        }
                        r_p4 = _mm256_hadd_pd( r_p4, r_p4 );

                        vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #else  // fallback code
            for( ; l <= o1_cols - block_size; l += block_size )
            {
                for( sz_t i = 0; i < o1_rows; i++ )
                {
                    const f3_t* vo1 = o1 + o1_stride * i + l;
                          f3_t* vr  = r  +  r_stride * i;

                    f3_t o1_p4[ block_size_p4 ][ 4 ];

                    for( sz_t k = 0; k < block_size_p4; k++ )
                    {
                        o1_p4[ k ][ 0 ] = vo1[ k * 4 + 0 ];
                        o1_p4[ k ][ 1 ] = vo1[ k * 4 + 1 ];
                        o1_p4[ k ][ 2 ] = vo1[ k * 4 + 2 ];
                        o1_p4[ k ][ 3 ] = vo1[ k * 4 + 3 ];
                    }

                    for( sz_t j = 0; j < o2_rows; j++ )
                    {
                        const f3_t* vo2 = o2 + j * o2_stride + l;

                        f3_t r_p4[ 4 ];
                        r_p4[ 0 ] = o1_p4[ 0 ][ 0 ] * vo2[ 0 ];
                        r_p4[ 1 ] = o1_p4[ 0 ][ 1 ] * vo2[ 1 ];
                        r_p4[ 2 ] = o1_p4[ 0 ][ 2 ] * vo2[ 2 ];
                        r_p4[ 3 ] = o1_p4[ 0 ][ 3 ] * vo2[ 3 ];

                        for( sz_t k = 1; k < block_size_p4; k++ )
                        {
                            r_p4[ 0 ] += o1_p4[ k ][ 0 ] * vo2[ k * 4 + 0 ];
                            r_p4[ 1 ] += o1_p4[ k ][ 1 ] * vo2[ k * 4 + 1 ];
                            r_p4[ 2 ] += o1_p4[ k ][ 2 ] * vo2[ k * 4 + 2 ];
                            r_p4[ 3 ] += o1_p4[ k ][ 3 ] * vo2[ k * 4 + 3 ];
                        }

                        r_p4[ 0 ] += r_p4[ 1 ];
                        r_p4[ 2 ] += r_p4[ 3 ];

                        vr[ j ] += r_p4[ 0 ] + r_p4[ 2 ];
                    }
                }
            }
        #endif // BMATH_AVX2

        for( sz_t i = 0; i < o1_rows; i++ )
        {
            const f3_t* vo1 = o1 + o1_stride * i;
                  f3_t* vr  = r  +  r_stride * i;
            for( sz_t j = 0; j < o2_rows; j++ )
            {
                const f3_t* vo2 = o2 + o2_stride * j;
                f3_t sum[ 4 ] = { 0, 0, 0, 0 };
                sz_t k = l;
                for( ; k <= o1_cols - 4; k += 4 )
                {
                    sum[ 0 ] += vo1[ k + 0 ] * vo2[ k + 0 ];
                    sum[ 1 ] += vo1[ k + 1 ] * vo2[ k + 1 ];
                    sum[ 2 ] += vo1[ k + 2 ] * vo2[ k + 2 ];
                    sum[ 3 ] += vo1[ k + 3 ] * vo2[ k + 3 ];
                }
                for( ; k < o1_cols; k++ )
                {
                    sum[ 0 ] += vo1[ k ] * vo2[ k ];
                }

                vr[ j ] += sum[ 0 ] + sum[ 1 ] + sum[ 2 ] + sum[ 3 ];
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_mul_htp2( const bmath_mf3_s* o, const bmath_mf3_s* op, bmath_mf3_s* res )
{
    if( res == o || res == op )
    {
        bmath_mf3_s* buf = bmath_mf3_s_create();
        bmath_mf3_s_set_size( buf, res->rows, res->cols );
        bmath_mf3_s_mul_htp2( o, op, buf );
        bmath_mf3_s_cpy( buf, res );
        bmath_mf3_s_discard( buf );
        return;
    }

    ASSERT(  o->cols ==  op->cols );
    ASSERT(  o->rows == res->rows );
    ASSERT( op->rows == res->cols );

    bmath_mf3_s_zro( res );

    bmath_mf3_s_mul_add_htp( o->data, o->stride, o->rows, o->cols, op->data, op->stride, op->rows, res->data, res->stride );
}

//---------------------------------------------------------------------------------------------------------------------



