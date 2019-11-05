/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_mf3.h"

#include "bmath_grt_f3.h"

#include "svd.h"

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_ubd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( a->rows < a->cols )
    {
        bmath_mf3_s_lbd2( u, a, v );
        bmath_mf3_s_lbd_to_ubd( u, a );
        return;
    }

    ASSERT( a->rows >= a->cols );

    /// at this point: a->rows >= a->cols;

    if( u )
    {
        bmath_mf3_s_one( u );
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
    }

    if( v )
    {
        bmath_mf3_s_one( v );
        ASSERT( v->cols == a->cols );
        ASSERT( v->rows == a->cols );
        ASSERT( v != a );
        ASSERT( v != u );
    }

    if( a->rows <= 1 )
    {
        if( u ) a->rows = u->cols;
        return; // nothing else to do
    }

    if( a->rows > a->cols * 4 )
    {
        if( u )
        {
            bmath_mf3_s_qrd( u, a );

            uz_t u_cols = u->cols;

            a->rows = a->cols;
            bmath_mf3_s* u2 = bmath_mf3_s_create();
            bmath_mf3_s_set_size( u2, a->rows, a->rows );
            bmath_mf3_s_ubd2( u2, a, v );
            u->cols = u2->rows;

            bmath_mf3_s_htp( u2, u2 );

            bmath_mf3_s_mul_htp( u, u2, u );
            bmath_mf3_s_discard( u2 );

            u->cols = u_cols;
            a->rows = u_cols;
        }
        else
        {
            bmath_mf3_s_qrd( NULL, a );
            uz_t a_rows = a->rows;
            a->rows = a->cols;
            bmath_mf3_s_ubd2( NULL, a, v );
            a->rows = a_rows;
        }
        return;
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( a->cols );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->cols; j++ )
    {
        // zero lower column
        for( uz_t l = a->rows - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ l * a->stride + j ] );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j, l, &gr, j + 1, a->cols );
        }

        // zero upper row
        for( uz_t l = a->cols - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j + 1 ], &a->data[ j * a->stride + l ] );
            if( v ) a->data[ j * a->stride + l ] = bmath_grt_f3_s_rho( &gr );
            grv.data[ l - 1 ] = gr;
        }

        bmath_mf3_s_sweep_dcol_rotate_rev( a, j + 1, a->cols - 1, &grv, j + 1, a->rows );
    }

    if( v ) // reverse construction of v
    {
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j + 1; k < a->cols - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j * a->stride + k + 1 ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( v, j + 1, k + 1, &gr, j, v->cols );
            }
        }
    }

    if( u ) // reverse construction of u
    {
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j, k + 1, &gr, j, u->cols );
            }
        }
        a->rows = u->cols;
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_lbd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( a->cols < a->rows )
    {
        bmath_mf3_s_ubd2( u, a, v );
        bmath_mf3_s_ubd_to_lbd( a, v );
        return;
    }

    /// at this point: a->cols >= a->rows;

    if( u )
    {
        bmath_mf3_s_one( u );
        ASSERT( u->cols == a->rows );
        ASSERT( u->rows == a->rows );
        ASSERT( u != a );
        ASSERT( u != v );
    }

    if( v )
    {
        bmath_mf3_s_one( v );
        ASSERT( v != a );
        ASSERT( v->rows == a->cols );
        ASSERT( v->cols == a->cols || v->cols == a->rows ); // v may be full or thin (nothing in-between)
    }

    if( a->cols <= 1 )
    {
        if( v ) a->cols = v->cols;
        return; // nothing else to do
    }

    if( a->cols > a->rows * 4 )
    {
        if( v )
        {
            bmath_mf3_s_lqd( a, v );

            uz_t v_cols = v->cols;

            a->cols = a->rows;
            bmath_mf3_s* v2 = bmath_mf3_s_create();
            bmath_mf3_s_set_size( v2, a->cols, a->cols );
            bmath_mf3_s_lbd2( u, a, v2 );
            v->cols = v2->rows;

            bmath_mf3_s_htp( v2, v2 );

            bmath_mf3_s_mul_htp( v, v2, v );
            bmath_mf3_s_discard( v2 );

            v->cols = v_cols;
            a->cols = v_cols;
        }
        else
        {
            bmath_mf3_s_lqd( a, NULL );
            uz_t a_cols = a->cols;
            a->cols = a->rows;
            bmath_mf3_s_lbd2( u, a, NULL );
            a->cols = a_cols;
        }
        return;
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( a->cols );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->rows; j++ )
    {
        // zero upper row
        for( uz_t l = a->cols - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ j * a->stride + l ] );
            if( v ) a->data[ j * a->stride + l ] = bmath_grt_f3_s_rho( &gr );
            grv.data[ l - 1 ] = gr;
        }

        bmath_mf3_s_sweep_dcol_rotate_rev( a, j, a->cols - 1, &grv, j + 1, a->rows );

        // zero lower column
        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ ( j + 1 ) * a->stride + j ], &a->data[ l * a->stride + j ] );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j + 1, l, &gr, j + 1, a->cols );
        }
    }

    if( v ) // reverse construction of v
    {
        for( uz_t j = a->rows - 1; j < a->rows; j-- )
        {
            for( uz_t k = j; k < a->cols - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j * a->stride + k + 1 ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( v, j, k + 1, &gr, j, v->cols );
            }
        }
        a->cols = v->cols;
    }

    if( u ) // reverse construction of u
    {
        for( uz_t j = a->rows - 2; j < a->rows; j-- )
        {
            for( uz_t k = j + 1; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j + 1, k + 1, &gr, j, u->cols );
            }
        }
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

static void qrd4_recursive_annihilate( bmath_mf3_s* a, uz_t offset, uz_t start, uz_t end )
{
    const sz_t block_size = 64;

    bmath_grt_f3_s gr;
    if( start + block_size < end )
    {
        uz_t mid = start + ( ( end - start ) >> 1 ) + 1;
        if( start < mid - 1 ) qrd4_recursive_annihilate( a, offset, start, mid - 1 );
        if( mid   < end     ) qrd4_recursive_annihilate( a, offset, mid, end );

        bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ start * a->stride + offset ], &a->data[ mid * a->stride + offset ] );
        a->data[ mid * a->stride + offset ] = bmath_grt_f3_s_rho( &gr );
        bmath_mf3_s_drow_rotate( a, start, mid, &gr, offset + 1, a->cols );
    }
    else
    {
        for( sz_t l = end; l > start; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ start * a->stride + offset ], &a->data[ l * a->stride + offset ] );
            a->data[ l * a->stride + offset ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, start, l, &gr, offset + 1, a->cols );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

static void qrd4_recursive_construct( bmath_mf3_s* u, const bmath_mf3_s* a, uz_t offset, uz_t start, uz_t end )
{
    const sz_t block_size = 64;

    bmath_grt_f3_s gr;
    if( start + block_size < end )
    {
        uz_t mid = start + ( ( end - start ) >> 1 ) + 1;

        f3_t rho = a->data[ offset + a->stride * mid ];
        bmath_grt_f3_s_init_from_rho( &gr, -rho );
        bmath_mf3_s_drow_rotate( u, start, mid, &gr, offset, u->cols );

        if( start < mid - 1 ) qrd4_recursive_construct( u, a, offset, start, mid - 1 );
        if( mid   < end     ) qrd4_recursive_construct( u, a, offset, mid, end );
    }
    else
    {
        for( sz_t k = start + 1; k <= end; k++ )
        {
            f3_t rho = a->data[ offset + a->stride * k ];
            bmath_grt_f3_s_init_from_rho( &gr, -rho );
            bmath_mf3_s_drow_rotate( u, start, k, &gr, offset, u->cols );
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

/** Using divide an conquer for more cache efficient interleaved progression
 */
void bmath_mf3_s_qrd4( bmath_mf3_s* u, bmath_mf3_s* a )
{
    if( u )
    {
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
        bmath_mf3_s_one( u );
    }

    if( a->rows <= 1 ) return; // nothing to do

    uz_t n = uz_min( a->cols, a->rows );

    for( uz_t j = 0; j < n; j++ )
    {
        qrd4_recursive_annihilate( a, j, j, a->rows - 1 );
    }

    if( u ) // reverse construction of u
    {
        for( uz_t j = n - 1; j < a->cols; j-- )
        {
            qrd4_recursive_construct( u, a, j, j, a->rows - 1 );
        }
        a->rows = u->cols;
    }

    // zero lower tridiagonal
    for( uz_t i = 1; i < a->rows; i++ )
    {
        f3_t* ai = a->data + i * a->stride;
        uz_t end = uz_min( a->cols, i );
        for( uz_t j = 0; j < end; j++ ) ai[ j ] = 0;
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_qrd_pmt4( bmath_mf3_s* u, bmath_mf3_s* a, bmath_pmt_s* p )
{
    if( p )
    {
        ASSERT( p->size == a->cols );
        bmath_pmt_s_one( p );
    }

    if( u )
    {
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
        bmath_mf3_s_one( u );
    }

    if( a->rows <= 1 ) return; // nothing to do

    uz_t n = uz_min( a->cols, a->rows );

    bmath_vf3_s* v = bmath_vf3_s_create();
    bmath_vf3_s_set_size( v, a->cols );
    bmath_vf3_s_zro( v );

    for( uz_t j = 0; j < a->rows; j++ )
    {
        f3_t* aj = a->data + j * a->stride;
        for( uz_t i = 0; i < a->cols; i++ ) v->data[ i ] += f3_sqr( aj[ i ] );
    }

    for( uz_t j = 0; j < n; j++ )
    {
        uz_t v_idx =  0;
        f3_t v_max = -1;
        for( uz_t i = j; i < a->cols; i++ )
        {
            v_idx = ( v->data[ i ] > v_max ) ? i : v_idx;
            v_max = ( v->data[ i ] > v_max ) ? v->data[ i ] : v_max;
        }

        f3_t_swap( v->data + j, v->data + v_idx );
        if( p ) uz_t_swap( p->data + j, p->data + v_idx );

        bmath_mf3_s_swap_col( a, j, v_idx );

        qrd4_recursive_annihilate( a, j, j, a->rows - 1 );

        f3_t* aj = a->data + j * a->stride;
        for( uz_t i = j; i < a->cols; i++ ) v->data[ i ] -= f3_sqr( aj[ i ] );
    }

    // make diagonal elements of a non-negative
    for( uz_t j = 0; j < n; j++ )
    {
        f3_t* aj = a->data + j * a->stride;
        if( aj[ j ] < 0 )
        {
            if( u ) u->data[ j * ( u->stride + 1 ) ] = -1;
            for( uz_t i = j; i < a->cols; i++ ) aj[ i ] = -aj[ i ];
        }
    }

    if( u ) // reverse construction of u
    {
        for( uz_t j = n - 1; j < a->cols; j-- )
        {
            qrd4_recursive_construct( u, a, j, j, a->rows - 1 );
        }
        a->rows = u->cols;
    }

    // zero lower tridiagonal
    for( uz_t i = 1; i < a->rows; i++ )
    {
        f3_t* ai = a->data + i * a->stride;
        uz_t end = uz_min( a->cols, i );
        for( uz_t j = 0; j < end; j++ ) ai[ j ] = 0;
    }

    bmath_vf3_s_discard( v );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_qrd_pmt2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_pmt_s* p )
{
    if( p )
    {
        ASSERT( p->size == a->cols );
        bmath_pmt_s_one( p );
    }

    if( u )
    {
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
    }

    if( a->rows <= 1 ) return; // nothing to do

    bmath_vf3_s* v = bmath_vf3_s_create();
    bmath_vf3_s_set_size( v, a->cols );
    bmath_vf3_s_zro( v );

    for( uz_t j = 0; j < a->rows; j++ )
    {
        f3_t* aj = a->data + j * a->stride;
        for( uz_t i = 0; i < a->cols; i++ ) v->data[ i ] += f3_sqr( aj[ i ] );
    }

    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->cols; j++ )
    {
        uz_t v_idx =  0;
        f3_t v_max = -1;
        for( uz_t i = j; i < a->cols; i++ )
        {
            v_idx = ( v->data[ i ] > v_max ) ? i : v_idx;
            v_max = ( v->data[ i ] > v_max ) ? v->data[ i ] : v_max;
        }

        f3_t_swap( v->data + j, v->data + v_idx );
        if( p ) uz_t_swap( p->data + j, p->data + v_idx );

        for( uz_t i = 0; i <= j; i++ )
        {
            f3_t* ai = a->data + i * a->stride;
            f3_t_swap( ai + j, ai + v_idx );
        }

        f3_t* aj = a->data + j * a->stride;

        // zero lower column
        for( uz_t l = a->rows - 1; l > j; l-- )
        {
            f3_t* al = a->data + l * a->stride;
            f3_t_swap( al + j, al + v_idx );

            bmath_grt_f3_s_init_and_annihilate_b( &gr, aj + j, al + j );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j, l, &gr, j + 1, a->cols );
        }

        for( uz_t i = j; i < a->cols; i++ ) v->data[ i ] -= f3_sqr( aj[ i ] );
    }

    if( u ) bmath_mf3_s_one( u );

    // make diagonal elements of a non-negative
    uz_t n = uz_min( a->cols, a->rows );
    for( uz_t j = 0; j < n; j++ )
    {
        f3_t* aj = a->data + j * a->stride;
        if( aj[ j ] < 0 )
        {
            if( u ) u->data[ j * ( u->stride + 1 ) ] = -1;
            for( uz_t i = j; i < a->cols; i++ ) aj[ i ] = -aj[ i ];
        }
    }

    if( u ) // reverse construction of u
    {
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j, k + 1, &gr, j, u->cols );
            }
        }
        a->rows = u->cols;
    }

    bmath_vf3_s_discard( v );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_trd2( bmath_mf3_s* a, bmath_mf3_s* v )
{
    ASSERT( bmath_mf3_s_is_hsm( a ) );

    uz_t n = a->rows;

    if( n <= 2 ) return; // nothing to do

    if( v )
    {
        ASSERT( v->cols == n );
        ASSERT( v->rows == n );
        ASSERT( v != a );
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( n );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < n; j++ )
    {
        // zero lower column
        for( uz_t k = n - 2; k > j; k-- )
        {
            uz_t l = k + 1;
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ k * a->stride + j ], &a->data[ l * a->stride + j ] );
            a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_arow_rotate( a, k, &gr, j + 1, l + 1 );
            f3_t* a00 = a->data + k * ( a->stride + 1 );
            a00[ -a->stride ] = a00[ -1 ];
            bmath_grt_f3_s_rotate( &gr, a00, a00 + 1 );
            grv.data[ l - 1 ] = gr;
        }

        if( bmath_arr_grt_f3_s_density( &grv, j + 1, n - 1 ) < 0.25 )
        {
            // sparse column operations
            for( uz_t k = n - 2; k > j; k-- )
            {
                gr = grv.data[ k ];
                bmath_mf3_s_acol_rotate( a, k, &gr, k + 1, n );
                if( k < n - 2 ) a->data[ ( k + 1 ) * ( a->stride + 1 ) + 1 ] = a->data[ ( k + 2 ) * ( a->stride + 1 ) - 1 ];
            }
        }
        else
        {
            // row swipes
            for( uz_t k = n - 1; k > j; k-- )
            {
                bmath_mf3_s_arow_swipe_rev( a, k, &grv, j + 1, k );
                if( k < n - 1 ) a->data[ k * ( a->stride + 1 ) + 1 ] = a->data[ ( k + 1 ) * ( a->stride + 1 ) - 1 ];
            }
        }
    }

    if( v )
    {
        bmath_mf3_s_one( v );
        for( uz_t j = 0; j < n; j++ )
        {
            for( uz_t l = n - 1; l > j + 1; l-- )
            {
                f3_t rho = a->data[ l * a->stride + j ];
                bmath_grt_f3_s_init_from_rho( &gr, rho );
                bmath_mf3_s_arow_rotate( v, l - 1, &gr, l - j - 1, n );
            }
        }
        bmath_mf3_s_htp( v, v );
    }

    // set off-tri diag elements zero
    for( uz_t i = 0; i < n; i++ )
    {
        f3_t* ai = a->data + i * a->stride;
        if( i > 1 ) for( uz_t j = 0; j < i - 1; j++ ) ai[ j ] = 0;
        for( uz_t j = i + 2; j < n; j++ ) ai[ j ] = 0;
    }

    // symmetrize off-diag
    for( uz_t i = 0; i < n - 1; i++ )
    {
        f3_t* ai = a->data + i * ( a->stride + 1 );
        f3_t b = ( ai[ 1 ] + ai[ a->stride ] ) * 0.5;
        ai[ 1 ] = ai[ a->stride ] = b;
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_decompose_qrd5( bmath_mf3_s* u, bmath_mf3_s* a )
{
    if( a->rows <= 1 ) return; // nothing to do

    if( u )
    {
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
    }

    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->cols; j++ )
    {
        // zero lower column
        for( uz_t l = a->rows - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ l * a->stride + j ] );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j, l, &gr, j + 1, a->cols );
        }
    }

    if( u ) // reverse construction of u
    {
        bmath_mf3_s_one( u );
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j, k + 1, &gr, j, u->cols );
            }
        }
        a->rows = u->cols;
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_decompose_lqd5( bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( a->cols <= 1 ) return; // nothing to do

    if( v )
    {
        ASSERT( v != a );
        ASSERT( v->rows == a->cols );
        ASSERT( v->cols == a->cols || v->cols == a->rows ); // v may be full or thin (nothing in-between)
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( a->cols );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->rows; j++ )
    {
        // zero upper row
        for( uz_t l = a->cols - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ j * a->stride + l ] );
            if( v ) a->data[ j * a->stride + l ] = bmath_grt_f3_s_rho( &gr );
            grv.data[ l - 1 ] = gr;
        }

        bmath_mf3_s_sweep_dcol_rotate_rev( a, j, a->cols - 1, &grv, j + 1, a->rows );
    }

    if( v ) // reverse construction of v
    {
        bmath_mf3_s_one( v );
        for( uz_t j = a->rows - 1; j < a->rows; j-- )
        {
            for( uz_t k = j; k < a->cols - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j * a->stride + k + 1 ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( v, j, k + 1, &gr, j, v->cols );
            }
        }
        a->cols = v->cols;
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

/// via distant givens
void bmath_mf3_s_decompose_ubd5( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( a->rows < a->cols )
    {
        bmath_mf3_s_decompose_lbd5( u, a, v );
        bmath_mf3_s_lbd_to_ubd( u, a );
        return;
    }

    ASSERT( a->rows >= a->cols );

    /// at this point: a->rows >= a->cols;

    if( a->rows <= 1 ) return; // nothing to do

    if( u )
    {
        ASSERT( u != a );
        ASSERT( u->rows == a->rows );
        ASSERT( u->cols == a->rows /*full*/ || u->cols == a->cols /*thin*/  ); // u may be full or thin (nothing in-between)
    }

    if( v )
    {
        ASSERT( v->cols == a->cols );
        ASSERT( v->rows == a->cols );
        ASSERT( v != a );
        ASSERT( v != u );
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( a->cols );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->cols; j++ )
    {
        // zero lower column
        for( uz_t l = a->rows - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ l * a->stride + j ] );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j, l, &gr, j + 1, a->cols );
        }

        // zero upper row
        for( uz_t l = a->cols - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j + 1 ], &a->data[ j * a->stride + l ] );
            if( v ) a->data[ j * a->stride + l ] = bmath_grt_f3_s_rho( &gr );
            grv.data[ l - 1 ] = gr;
        }

        bmath_mf3_s_sweep_dcol_rotate_rev( a, j + 1, a->cols - 1, &grv, j + 1, a->rows );
    }

    if( v ) // reverse construction of v
    {
        bmath_mf3_s_one( v );
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j + 1; k < a->cols - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j * a->stride + k + 1 ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( v, j + 1, k + 1, &gr, j, v->cols );
            }
        }
    }

    if( u ) // reverse construction of u
    {
        bmath_mf3_s_one( u );
        for( uz_t j = a->cols - 1; j < a->cols; j-- )
        {
            for( uz_t k = j; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j, k + 1, &gr, j, u->cols );
            }
        }
        a->rows = u->cols;
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_decompose_lbd5( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( a->cols < a->rows )
    {
        bmath_mf3_s_decompose_ubd5( u, a, v );
        bmath_mf3_s_ubd_to_lbd( a, v );
        return;
    }

    /// at this point: a->cols >= a->rows;

    if( a->cols <= 1 ) return; // nothing to do

    if( u )
    {
        ASSERT( u->cols == a->rows );
        ASSERT( u->rows == a->rows );
        ASSERT( u != a );
        ASSERT( u != v );
    }

    if( v )
    {
        ASSERT( v != a );
        ASSERT( v->rows == a->cols );
        ASSERT( v->cols == a->cols || v->cols == a->rows ); // v may be full or thin (nothing in-between)
    }

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( a->cols );
    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < a->rows; j++ )
    {
        // zero upper row
        for( uz_t l = a->cols - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ j * a->stride + j ], &a->data[ j * a->stride + l ] );
            if( v ) a->data[ j * a->stride + l ] = bmath_grt_f3_s_rho( &gr );
            grv.data[ l - 1 ] = gr;
        }

        bmath_mf3_s_sweep_dcol_rotate_rev( a, j, a->cols - 1, &grv, j + 1, a->rows );

        // zero lower column
        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gr, &a->data[ ( j + 1 ) * a->stride + j ], &a->data[ l * a->stride + j ] );
            if( u ) a->data[ l * a->stride + j ] = bmath_grt_f3_s_rho( &gr );
            bmath_mf3_s_drow_rotate( a, j + 1, l, &gr, j + 1, a->cols );
        }
    }

    if( v ) // reverse construction of v
    {
        bmath_mf3_s_one( v );
        for( uz_t j = a->rows - 1; j < a->rows; j-- )
        {
            for( uz_t k = j; k < a->cols - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j * a->stride + k + 1 ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( v, j, k + 1, &gr, j, v->cols );
            }
        }
        a->cols = v->cols;
    }

    if( u ) // reverse construction of u
    {
        bmath_mf3_s_one( u );
        for( uz_t j = a->rows - 2; j < a->rows; j-- )
        {
            for( uz_t k = j + 1; k < a->rows - 1; k++ )
            {
                f3_t rho = 0;
                f3_t_swap( &a->data[ j + a->stride * ( k + 1 ) ], &rho );
                bmath_grt_f3_s_init_from_rho( &gr, -rho );
                bmath_mf3_s_drow_rotate( u, j + 1, k + 1, &gr, j, u->cols );
            }
        }
    }

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_ubd_to_lbd4( bmath_mf3_s* a, bmath_mf3_s* v )
{
    uz_t n = uz_min( a->rows, a->cols );

    if( n <= 1 ) return; // nothing to do

    if( v ) ASSERT( v->cols >= n );

    bmath_arr_grt_f3_s grv = bmath_arr_grt_f3_of_size( n );

    for( uz_t j = 0; j < n - 1; j++ )
    {
        f3_t* aj = a->data + j * ( a->stride + 1 );
        bmath_grt_f3_s_init_and_annihilate_b( &grv.data[ j ], aj, aj + 1 );
        bmath_grt_f3_s_rotate( &grv.data[ j ], aj + a->stride, aj + a->stride + 1 );
    }

    if( v ) bmath_mf3_s_sweep_acol_rotate_fwd( v, 0, n - 1, &grv, 0, v->rows );

    bmath_arr_grt_f3_s_down( &grv );
}

//----------------------------------------------------------------------------------------------------------------------

// v transposed
void bmath_mf3_s_ubd_to_lbd4_htp( bmath_mf3_s* a, bmath_mf3_s* v )
{
    uz_t n = uz_min( a->rows, a->cols );

    if( n <= 1 ) return; // nothing to do

    if( v ) ASSERT( v->rows >= n );

    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < n - 1; j++ )
    {
        f3_t* aj = a->data + j * ( a->stride + 1 );
        bmath_grt_f3_s_init_and_annihilate_b( &gr, aj, aj + 1 );
        bmath_grt_f3_s_rotate( &gr, aj + a->stride, aj + a->stride + 1 );
        if( v ) bmath_mf3_s_arow_rotate( v, j, &gr, 0, v->cols );
    }
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_lbd_to_ubd4( bmath_mf3_s* u, bmath_mf3_s* a )
{
    uz_t n = uz_min( a->rows, a->cols );

    if( n <= 1 ) return; // nothing to do

    if( u ) ASSERT( u->cols >= n );

    bmath_arr_grt_f3_s gru = bmath_arr_grt_f3_of_size( n );

    for( uz_t j = 0; j < n - 1; j++ )
    {
        f3_t* aj = a->data + j * ( a->stride + 1 );
        bmath_grt_f3_s_init_and_annihilate_b( &gru.data[ j ], aj, aj + a->stride );
        bmath_grt_f3_s_rotate( &gru.data[ j ], aj + 1, aj + a->stride + 1 );
    }

    if( u ) bmath_mf3_s_sweep_acol_rotate_fwd( u, 0, n - 1, &gru, 0, u->rows );

    bmath_arr_grt_f3_s_down( &gru );
}

//----------------------------------------------------------------------------------------------------------------------

// u transposed
void bmath_mf3_s_lbd_to_ubd4_htp( bmath_mf3_s* u, bmath_mf3_s* a )
{
    uz_t n = uz_min( a->rows, a->cols );

    if( n <= 1 ) return; // nothing to do

    if( u ) ASSERT( u->rows >= n );

    bmath_grt_f3_s gr;

    for( uz_t j = 0; j < n - 1; j++ )
    {
        f3_t* aj = a->data + j * ( a->stride + 1 );
        bmath_grt_f3_s_init_and_annihilate_b( &gr, aj, aj + a->stride );
        bmath_grt_f3_s_rotate( &gr, aj + 1, aj + a->stride + 1 );
        if( u ) bmath_mf3_s_arow_rotate( u, j, &gr, 0, u->cols );
    }
}

//----------------------------------------------------------------------------------------------------------------------

void reconstruct_htp( const bmath_mf3_s* u, const bmath_mf3_s* a, const bmath_mf3_s* v, bmath_mf3_s* res )
{
    if( !u || !v ) return;
    bmath_mf3_s* ut = bmath_mf3_s_clone( u );
    bmath_mf3_s_htp( u, ut );
    bmath_mf3_s_mul( a, v, res );
    bmath_mf3_s_mul( ut, res, res );
    bmath_mf3_s_discard( ut );
}

bl_t ud_is_zero( bmath_mf3_s* a, uz_t row_idx )
{
    f3_t* a00 = a->data + ( a->stride + 1 ) * row_idx;
    f3_t  a01_abs = f3_abs( a00[ 1 ] );
    if( a01_abs < f3_lim_min ||
        (
            a01_abs < f3_abs( a00[ 0             ] ) * f3_lim_eps &&
            a01_abs < f3_abs( a00[ a->stride + 1 ] ) * f3_lim_eps )
        )
    {
        a00[ 1 ] = 0;
        return true;
    }
    return false;
}

void ud_set_zero( bmath_mf3_s* a, uz_t row_idx )
{
    a->data[ ( a->stride + 1 ) * row_idx + 1 ] = 0;
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_decompose_lbd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    ASSERT( a->cols >= a->rows );
    if( a->cols <= 1 ) return; // nothing to do

    if( u )
    {
        ASSERT( u->cols == a->rows );
        ASSERT( u->rows == a->rows );
        ASSERT( u != a );
        ASSERT( u != v );
        bmath_mf3_s_one( u );
    }

    if( v )
    {
        ASSERT( v->cols == a->cols );
        ASSERT( v->rows == a->cols );
        ASSERT( v != a );
        bmath_mf3_s_one( v );
    }

    bmath_grt_f3_s* gru = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, a->rows, NULL ); // left rotators
    bmath_grt_f3_s* grv = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, a->cols, NULL ); // right rotators

    /* Explicitly zeroing the rotator field entices LFU cache
       to favor this speed critical memory buffer.
    */
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), gru, a->rows );
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), grv, a->cols );

    for( uz_t j = 0; j < a->rows; j++ )
    {
        // zero upper row
        for( uz_t l = a->cols - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &grv[ l ], a->data + j * a->stride + l - 1, a->data + j * a->stride + l );
        }

        for( uz_t i = j + 1; i < a->rows; i++ )
        {
            f3_t* ai = a->data + i * a->stride;
            for( uz_t l = a->cols - 1; l > j; l-- )
            {
                bmath_grt_f3_s_rotate( &grv[ l ], ai + l - 1, ai + l );
            }
        }

        // zero lower column
        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gru[ l ], a->data + ( l - 1 ) * a->stride + j, a->data + l * a->stride + j );
        }

        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            f3_t* row = a->data + ( l - 1 ) * a->stride;
            bmath_grt_f3_s_row_rotate( &gru[ l ], row, row + a->stride, j + 1, a->cols );
        }

        if( v )
        {
            for( uz_t l = a->cols - 1; l > j; l-- )
            {
                f3_t* row = v->data + ( l - 1 ) * v->stride;
                bmath_grt_f3_s_row_rotate( &grv[ l ], row, row + v->stride, l - 1 - j, v->cols );
            }
        }

        if( u )
        {
            for( uz_t l = a->rows - 1; l > j + 1; l-- )
            {
                f3_t* row = u->data + ( l - 1 ) * u->stride;
                bmath_grt_f3_s_row_rotate( &gru[ l ], row, row + u->stride, l - 1 - j, u->cols );
            }
        }
    }


    bcore_free( gru );
    bcore_free( grv );
}

//----------------------------------------------------------------------------------------------------------------------

void bmath_mf3_s_decompose_ubd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    if( u )
    {
        ASSERT( u->cols == a->rows );
        ASSERT( u->rows == a->rows );
        ASSERT( u != a );
        ASSERT( u != v );
        bmath_mf3_s_one( u );
    }

    if( v )
    {
        ASSERT( v->cols == a->cols );
        ASSERT( v->rows == a->cols );
        ASSERT( v != a );
        bmath_mf3_s_one( v );
    }

    bmath_grt_f3_s gr;
    uz_t n = uz_min( a->cols, a->rows );
    uz_t m = uz_max( a->cols, a->rows );

    if( m <= 1 ) return; // nothing to do

    bmath_grt_f3_s* gru = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, a->rows, NULL ); // left rotators
    bmath_grt_f3_s* grv = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, a->cols, NULL ); // right rotators

    /* Explicitly zeroing the rotator field entices LFU cache
       to favor this speed critical memory buffer.
    */
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), gru, a->rows );
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), grv, a->cols );

    for( uz_t j = 0; j < n; j++ )
    {
        // zero upper row
        for( uz_t l = a->cols - 1; l > j; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &grv[ l ], a->data + j * a->stride + l - 1, a->data + j * a->stride + l );
        }

        for( uz_t i = j + 1; i < a->rows; i++ )
        {
            f3_t* ai = a->data + i * a->stride;
            for( uz_t l = a->cols - 1; l > j; l-- )
            {
                bmath_grt_f3_s_rotate( &grv[ l ], ai + l - 1, ai + l );
            }
        }

        // zero lower column
        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            bmath_grt_f3_s_init_and_annihilate_b( &gru[ l ], a->data + ( l - 1 ) * a->stride + j, a->data + l * a->stride + j );
        }

        for( uz_t l = a->rows - 1; l > j + 1; l-- )
        {
            f3_t* row = a->data + ( l - 1 ) * a->stride;
            bmath_grt_f3_s_row_rotate( &gru[ l ], row, row + a->stride, j + 1, a->cols );
        }

        if( v )
        {
            for( uz_t l = a->cols - 1; l > j; l-- )
            {
                f3_t* row = v->data + ( l - 1 ) * v->stride;
                bmath_grt_f3_s_row_rotate( &grv[ l ], row, row + v->stride, l - 1 - j, v->cols );
            }
        }

        if( u )
        {
            for( uz_t l = a->rows - 1; l > j + 1; l-- )
            {
                f3_t* row = u->data + ( l - 1 ) * u->stride;
                bmath_grt_f3_s_row_rotate( &gru[ l ], row, row + u->stride, l - 1 - j, u->cols );
            }
        }
    }

    // flip over left lower off-diagonal element
    for( uz_t k = 0; k < uz_min( n, a->rows - 1 ); k++ )
    {
        f3_t* ak = a->data + k * ( a->stride + 1 );
        f3_t* al = ak + a->stride;
        bmath_grt_f3_s_init_and_annihilate_b( &gr, ak, al );
        bmath_grt_f3_s_rotate( &gr, ak + 1, al + 1 );
        if( u ) bmath_grt_f3_s_row_rotate( &gr, u->data + k * u->stride, u->data + ( k + 1 ) * u->stride, 0, u->cols );
    }

    bcore_free( gru );
    bcore_free( grv );
}

//----------------------------------------------------------------------------------------------------------------------

bl_t bmath_mf3_s_svd_full( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    // creating upper-bidiagonal
    bmath_mf3_s_decompose_ubd2( u, a, v );

    uz_t n = f3_min( a->cols, a->rows );
    if( n <= 1 ) return true; // nothing else to do

    bmath_grt_f3_s* gru = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, n, NULL ); // left rotators
    bmath_grt_f3_s* grv = bcore_un_alloc( sizeof( bmath_grt_f3_s ), NULL, 0, n, NULL ); // right rotators

    /* Explicitly zeroing the rotator field entices LFU cache
       to favor this speed critical memory buffer.
    */
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), gru, n );
    bcore_u_memzero( sizeof( bmath_grt_f3_s ), grv, n );

    bl_t success = true;

    // in practice convergence hardly ever needs more than 4 cycles
    const uz_t max_cyles = 50;

    for( uz_t block_n = n; block_n > 1; block_n-- )
    {
        uz_t cycle;
        for( cycle = 0; cycle < max_cyles; cycle++ )
        {
            f3_t lambda;

            {
                f3_t* a0 = a->data + ( a->stride + 1 ) * ( block_n - 2 );
                f3_t* a1 = a0 + a->stride;

                // exit cycle if bottom off-diagonal is zero
                if( ( f3_abs( a0[ 1 ] ) < f3_lim_min ) || ( f3_abs( a0[ 1 ] ) < f3_abs( a1[ 1 ] ) * f3_lim_eps ) )
                {
                    a0[ 1 ] = 0;
                    break;
                }

                f3_t m11 = f3_sqr( a1[ 1 ] );
                f3_t m00 = f3_sqr( a0[ 0 ] ) + f3_sqr( a0[ 1 ] );
                f3_t m01 = a0[ 1 ] * a1[ 1 ];

                f3_t p = 0.5 * ( m00 + m11 );
                f3_t d = sqrt( 0.25 * f3_sqr( m00 - m11 ) + m01 * m01 );

                // set shift to eigenvalue of lower 2x2 sub-matrix which is closest to m11
                lambda = ( m11 >= p ) ? p + d : p - d;
            }

            f3_t* a0 = a->data;
            f3_t* a1 = a->data + a->stride;

            bmath_grt_f3_s gr_l; // left rotation
            bmath_grt_f3_s gr_r; // right rotation

            // left rotation strategically diagonalizes a * aT but creates a1[ 0 ]
            bmath_grt_f3_s_init_to_annihilate_b( &gr_l, f3_sqr( a0[ 0 ] ) + f3_sqr( a0[ 1 ] ) - lambda, a0[ 1 ] * a1[ 1 ] );

            // right rotation annihilates a1[0]
            bmath_grt_f3_s_init_to_annihilate_b( &gr_r, gr_l.c * a1[ 1 ] - gr_l.s * a0[ 1 ], gr_l.s * a0[ 0 ] );

            f3_t a00 = a0[ 0 ];
            f3_t a01 = a0[ 1 ];
            f3_t a11 = a1[ 1 ];

            a0[ 0 ] = gr_r.c * a00 + gr_r.s * a01; a0[ 1 ] = gr_r.c * a01 - gr_r.s * a00;
            a1[ 0 ] =                gr_r.s * a11; a1[ 1 ] = gr_r.c * a11;
            bmath_grt_f3_s_row_rotate( &gr_l, a0, a1, 0, 3 );
            a1[ 0 ] = 0;

            gru[ 0 ] = gr_l;
            grv[ 0 ] = gr_r;

            // chasing, annihilating off-bidiagonals
            for( uz_t k = 0; k < block_n - 2; k++ )
            {
                f3_t* ak = a->data + ( a->stride + 1 ) * k + 1;
                f3_t* al = ak + a->stride;
                f3_t* am = al + a->stride;
                bmath_grt_f3_s_init_and_annihilate_b( &gr_r, ak, ak + 1 );
                bmath_grt_f3_s_rotate( &gr_r, al, al + 1 );
                bmath_grt_f3_s_rotate( &gr_r, am, am + 1 );

                bmath_grt_f3_s_init_and_annihilate_b( &gr_l, al, am );
                bmath_grt_f3_s_rotate( &gr_l, al + 1, am + 1 );
                if( k < block_n - 3 ) bmath_grt_f3_s_rotate( &gr_l, al + 2, am + 2 );

                gru[ k + 1 ] = gr_l;
                grv[ k + 1 ] = gr_r;
            }

            if( u )
            {
                for( uz_t i = 0; i < block_n - 1; i++ )
                {
                    bmath_grt_f3_s_row_rotate( &gru[ i ], u->data + u->stride * i, u->data + u->stride * ( i + 1 ), 0, u->cols );
                }
            }

            if( v )
            {
                for( uz_t i = 0; i < block_n - 1; i++ )
                {
                    bmath_grt_f3_s_row_rotate( &grv[ i ], v->data + v->stride * i, v->data + v->stride * ( i + 1 ), 0, v->cols );
                }
            }
        }
        if( cycle == max_cyles ) success = false;
    }

    bcore_free( gru );
    bcore_free( grv );

    if( !success ) return false;

    // sorts by descending diagonal values; turns negative values
    for( uz_t i = 0; i < n; i++ )
    {
        f3_t vmax = f3_abs( a->data[ i * ( a->stride + 1 ) ] );
        uz_t imax = i;
        for( uz_t j = i + 1; j < n; j++ )
        {
            f3_t v = f3_abs( a->data[ j * ( a->stride + 1 ) ] );
            imax = ( v > vmax ) ? j : imax;
            vmax = ( v > vmax ) ? v : vmax;
        }
        if( imax != i )
        {
            f3_t_swap( a->data + i * ( a->stride + 1 ), a->data + imax * ( a->stride + 1 ) );
            if( u ) bmath_mf3_s_swap_row( u, i, imax );
            if( v ) bmath_mf3_s_swap_row( v, i, imax );
        }

        if( a->data[ i * ( a->stride + 1 ) ] < 0 )
        {
            a->data[ i * ( a->stride + 1 ) ] *= -1.0;
            if( v ) bmath_mf3_s_mul_fx_to_row( v, -1.0, i );
        }
    }

    return success;
}

//----------------------------------------------------------------------------------------------------------------------

/// full svd when u, v are fully allocated, thin svd otherwise
bl_t bmath_mf3_s_svd2( bmath_mf3_s* u, bmath_mf3_s* a, bmath_mf3_s* v )
{
    ASSERT( u != a && v != a );
    ASSERT( u != v || u == NULL || v == NULL );

    // on square matrices thin == full
    if( a->rows == a->cols )
    {
        return bmath_mf3_s_svd_full( u, a, v );
    }

    bl_t success = true;

    bmath_mf3_s* q = ( a->rows < a->cols ) ? u : v;
    bmath_mf3_s* w = ( a->rows < a->cols ) ? v : u;

    if( a->rows < a->cols ) // v should be thin or NULL
    {
        if( !v || v->rows == a->cols ) // v is NULL or full
        {
            return bmath_mf3_s_svd_full( u, a, v );
        }

        // assert v being thin
        ASSERT( v->cols == a->cols );
        ASSERT( v->rows == a->rows );

        // v = a
        bmath_mf3_s_cpy( a, v );
        success = bmath_mf3_s_svd_full( u, a, NULL );
        a->cols = a->rows;
    }
    else // a->rows > a->cols
    {
        if( !u || u->rows == a->rows ) // u is NULL or full
        {
            return bmath_mf3_s_svd_full( u, a, v );
        }

        // assert u being thin
        ASSERT( u->rows == a->cols );
        ASSERT( u->cols == a->rows );

        // u = aT
        bmath_mf3_s_htp( a, u );
        success = bmath_mf3_s_svd_full( NULL, a, v );
        a->rows = a->cols;
    }

    bmath_vf3_s* buf = bmath_vf3_s_create();
    bmath_vf3_s_set_size( buf, w->rows );

    // reconstruct thin matrix and strengthen orthogonality via modified gram schmidt

    // w = q * w
    for( uz_t i = 0; i < w->cols; i++ )
    {
        bmath_mf3_s_get_col_vec( w, i, buf );
        for( uz_t j = 0; j < q->rows; j++ )
        {
            bmath_vf3_s qj = bmath_mf3_s_get_row_weak_vec( q, j );
            w->data[ j * w->stride + i ] = bmath_vf3_s_f3_mul_vec( &qj, buf );
        }
    }

    bmath_vf3_s_discard( buf );

    // re-orthonormalize w to reduce the effect of rounding errors and deal with (near) zero singular values
    for( uz_t i = 0; i < w->rows; i++ )
    {
        bmath_vf3_s wi = bmath_mf3_s_get_row_weak_vec( w, i );
        for( uz_t loop = 0; loop < 2; loop++ )
        {
            for( uz_t j = 0; j < i; j++ )
            {
                bmath_vf3_s wj = bmath_mf3_s_get_row_weak_vec( w, j );
                f3_t f = bmath_vf3_s_f3_mul_vec( &wi, &wj );
                for( uz_t k = 0; k < wj.size; k++ ) wi.data[ k ] -= wj.data[ k ] * f;
            }

            f3_t w_sqr = bmath_vf3_s_f3_sqr( &wi );

            // TODO: analyze if we need to raise that limit and/or make it dependent on SV distribution
            if( w_sqr > f3_lim_min )
            {
                // normalize wi and exit 'loop'
                bmath_vf3_s_mul_f3( &wi, 1.0 / f3_srt( w_sqr ), &wi );
                break;
            }
            else // initialize w with the most orthogonal unit vector and repeat
            {
                bmath_vf3_s_zro( &wi );
                for( uz_t j = 0; j < i; j++ )
                {
                    bmath_vf3_s wj = bmath_mf3_s_get_row_weak_vec( w, j );
                    for( uz_t k = 0; k < wi.size; k++ ) wi.data[ k ] += f3_sqr( wj.data[ k ] );
                }
            }
            uz_t idx_min = bmath_vf3_s_idx_min( &wi );
            bmath_vf3_s_zro( &wi );
            wi.data[ idx_min ] = 1;
        }
    }

    return success;
}

//----------------------------------------------------------------------------------------------------------------------



