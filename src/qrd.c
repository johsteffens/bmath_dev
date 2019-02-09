/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "qrd.h"

//----------------------------------------------------------------------------------------------------------------------

// plain qr of a square nxn matrix a = q^T * a' with q, a indexed
// q must be externally initialized
void bmath_plain_htp_qrd_xqa( const sz_t* m_x, f3_t* q, sz_t q_s, f3_t* a, sz_t a_s, sz_t n )
{
    if( n <= 1 ) return; // nothing else to do

    bmath_grt_f3_s gr;

    for( sz_t j = 0; j < n; j++ )
    {
        f3_t* aj = a + m_x[ j ] * a_s + j;
        f3_t* qj = q + m_x[ j ] * q_s    ;

        // zero lower column
        for( uz_t l = n - 1; l > j; l-- )
        {
            f3_t* al = a + m_x[ l ] * a_s + j;
            f3_t* ql = q + m_x[ l ] * q_s;
            bmath_grt_f3_s_init_and_annihilate_b( &gr, aj, al );
            bmath_simd_f3_row_rotate( aj + 1, al + 1, n - j - 1, &gr );
            bmath_simd_f3_row_rotate( qj    , ql    , n,         &gr );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

// plain qr of a square nxn matrix a = q^T * a' with a indexed
// q internally initialized
static void bmath_plain_htp_qrd_xa( const sz_t* m_x, f3_t* q, sz_t q_s, f3_t* a, sz_t a_s, sz_t n )
{
    if( n <= 1 ) return; // nothing else to do

    bmath_grt_f3_s gr;

    for( sz_t i = 0; i < n; i++ )
    {
        for( sz_t j = 0; j < n; j++ ) q[ i * q_s + j ] = ( i == j ) ? 1 : 0;
    }

    sz_t n_mul = 0;

    for( sz_t j = 0; j < n; j++ )
    {
        // zero lower column
        for( sz_t k = n - 2; k >= j; k-- )
        {
            f3_t* ak = a + m_x[ ( k + 0 ) ] * a_s + j;
            f3_t* qk = q +      ( k + 0 )   * q_s;
            f3_t* al = a + m_x[ ( k + 1 ) ] * a_s + j;
            f3_t* ql = q +      ( k + 1 )   * q_s;
            bmath_grt_f3_s_init_and_annihilate_b( &gr, ak, al );
            bmath_simd_f3_row_rotate( ak + 1,     al + 1,     n - j - 1, &gr );
            bmath_simd_f3_row_rotate( qk + k - j, ql + k - j, n - k + j, &gr );

            n_mul += 4 * ( n - j );
            n_mul += 4 * ( n - k + j );
        }
    }
    bcore_msg_fa( "qrd2: #<sz_t>\n", n_mul );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_qrd2( bmath_mf3_s* q, bmath_mf3_s* a )
{
    ASSERT( a );
    ASSERT( q );
    ASSERT( a->rows == a->cols );
    ASSERT( a->rows == q->rows );
    ASSERT( a->cols == q->cols );

    bmath_mf3_s_one( q );

    bcore_arr_sz_s* m_x = bcore_arr_sz_s_create();
//    bcore_arr_sz_s_step_fill( m_x, a->rows - 1, -1, a->rows );

    bcore_arr_sz_s_step_fill( m_x, 0, 1, a->rows );

    bmath_plain_htp_qrd_xa( m_x->data, q->data, q->stride, a->data, a->stride, a->rows );
    bcore_arr_sz_s_discard( m_x );

    bmath_mf3_s_htp( q, q );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_qrd3( bmath_mf3_s* q, bmath_mf3_s* a )
{
    ASSERT( a );
    ASSERT( q );
    ASSERT( a->rows == a->cols );
    ASSERT( a->rows == q->rows );
    ASSERT( a->cols == q->cols );

    sz_t n_mul = 0;

    bmath_mf3_s_one( q );
    bmath_grt_f3_s gr;

    sz_t n = a->rows;

    for( sz_t k = 1; k < n; k++ )
    {
        f3_t* a0 = a->data + ( n - k - 1 ) * a->stride;
        f3_t* q0 = q->data + ( n - k - 1 ) * q->stride + n - k - 1;

        // use first row (offs 1) of q as buffer for last a-column
        for( sz_t j = 1; j <= k; j++ ) q0[ j ] = a0[ j * a->stride + k ];

        for( sz_t i = 1; i <= k; i++ )
        {
            f3_t* qi = q0 + i * q->stride;
            f3_t v = 0;
            for( sz_t j = 1; j <= k; j++ )
            {
                v += qi[ j ] * q0[ j ];
                n_mul++;
            }
            a0[ i * a->stride + k ] = v;
        }

        for( sz_t j = 1; j <= k; j++ ) q0[ j ] = 0; // reset q0

        for( sz_t i = 0; i < k; i++ )
        {
            f3_t* ai = a0 + i * ( a->stride + 1 );
            f3_t* qi = q0 + i * ( q->stride     );

            bmath_grt_f3_s_init_and_annihilate_b( &gr, ai, ai + a->stride );
            bmath_simd_f3_row_rotate( ai + 1, ai + a->stride + 1, k - i, &gr );
            bmath_simd_f3_row_rotate( qi    , qi + q->stride    , k + 1, &gr );
            n_mul += 4 * ( k - i + 1 );
        }
        n_mul += ( k + 1 ) * ( k + 1 ) * 4;
    }

    bcore_msg_fa( "qrd3: #<sz_t>\n", n_mul );

    bmath_mf3_s_htp( q, q );
}

//----------------------------------------------------------------------------------------------------------------------

