/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <sys/time.h>

#include "bmath_std.h"
#include "bmath_mfx_eval.h"

#include "svd.h"
#include "mul.h"
#include "smul.h"
#include "qrd.h"
#include "cnn.h"
#include "snn.h"
#include "zorder.h"

//----------------------------------------------------------------------------------------------------------------------

bl_t ltr_inv( const bmath_mf3_s* o, bmath_mf3_s* res )
{
    // Algorithm works in-place: No need to check for o == res;
    ASSERT( BCATU(bmath_mf3_s,is_square)( o ) );
    ASSERT( BCATU(bmath_mf3_s,is_equ_size)( o, res ) );

    BCATU(bmath_mf3_s,cpy)( o, res );

    sz_t n = res->rows;

    bl_t success = true;

    bmath_vf3_s* buf = BCATU(bmath_vf3_s,create)();
    BCATU(bmath_vf3_s,set_size)( buf, n );
    f3_t* q = buf->data;

    for( sz_t i = 0; i < n; i++ )
    {
        f3_t* vi = res->data + i * res->stride;

        for( sz_t j = 0; j < i; j++ ) q[ j ] = 0;

        for( sz_t j = 0; j < i; j++ )
        {
            f3_t* vj = res->data + j * res->stride;
            for( sz_t k = 0; k <= j; k++ ) q[ k ] -= vi[ j ] * vj[ k ];
        }

        bl_t stable = ( BCATU(f3,abs)( vi[ i ] ) > BCATU(f3,lim_min) );
        vi[ i ] = stable ? 1.0 / vi[ i ] : 0;
        success = success && stable;

        for( sz_t j = 0; j < i; j++ ) vi[ j ] = q[ j ] * vi[ i ];
    }

    BCATU(bmath_vf3_s,discard)( buf );

    return success;
}

//----------------------------------------------------------------------------------------------------------------------

void test_ltr_inv( sz_t n, bl_t (*fp_ltr_inv)( const bmath_mf3_s* o, bmath_mf3_s* res ) )
{
    BLM_INIT();

    bmath_mf3_s* m1 = BLM_CREATE( bmath_mf3_s );
    bmath_mf3_s* m2 = BLM_CREATE( bmath_mf3_s );
    bmath_mf3_s* m3 = BLM_CREATE( bmath_mf3_s );

    bmath_mf3_s_set_size( m1, n, n );
    bmath_mf3_s_set_size( m2, n, n );
    bmath_mf3_s_set_size( m3, n, n );

    bmath_mf3_s_set_random( m1, true, true, 0, 1.0, -1.0, 1.0, NULL );

    // set m1 lower triangular
    for( sz_t i = 0; i < n; i++ ) for( sz_t j = i + 1; j < n; j++ ) m1->data[ i * m1->stride + j ] = 0;

    CPU_TIME_TO_STDOUT( fp_ltr_inv( m1, m2 ) );
    bmath_mf3_s_mul( m1, m2, m3 );

    ASSERT( bmath_mf3_s_is_near_one( m3, 1E-8 ) );

    BLM_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

bl_t ltr_inv_htp( const bmath_mf3_s* o, bmath_mf3_s* res )
{
    bl_t success = ltr_inv( o, res );
    sz_t n = res->rows;

    // transpose result
    for( sz_t i = 0; i < n; i++ )
    {
        f3_t* vi = res->data + i * res->stride;
        for( sz_t j = 0; j < i; j++ )
        {
            f3_t* vj = res->data + j * res->stride;
            vj[ i ] = vi[ j ];
            vi[ j ] = 0;
        }
    }


//    // Algorithm works in-place: No need to check for o == res;
//    ASSERT( BCATU(bmath_mf3_s,is_square)( o ) );
//    ASSERT( BCATU(bmath_mf3_s,is_equ_size)( o, res ) );
//
//    BCATU(bmath_mf3_s,cpy)( o, res );
//
//    sz_t n = res->rows;
//    bl_t success = true;
//
//    // inverting diagonal elements
//    for( sz_t i = 0; i < n; i++ )
//    {
//        f3_t* v = res->data + i * ( res->stride + 1 );
//        if( BCATU(f3,abs)( *v ) < BCATU(f3,lim_min) ) success = false;
//        *v = ( *v != 0 ) ? 1.0 / *v : 0;
//    }
//
//    // upper off-diagonal elements
//    // TODO: work out a solution with innermost read-write stream (rather than read-read-accu stream)
//    for( sz_t i = 0; i < n; i++ )
//    {
//        f3_t* vi = res->data + i * res->stride;
//        for( sz_t j = i + 1; j < n; j++ )
//        {
//            f3_t* vj = res->data + j * res->stride;
//            vi[ j ] = -vj[ j ] * BCATU(bmath,f3,t_vec,mul_vec)( vj + i, vi + i, j - i );
//            vj[ i ] = 0;
//        }
//    }

    return success;
}

//----------------------------------------------------------------------------------------------------------------------

void test_ltr_inv_htp( sz_t n, bl_t (*fp_ltr_inv_htp)( const bmath_mf3_s* o, bmath_mf3_s* res ) )
{
    BLM_INIT();

    bmath_mf3_s* m1 = BLM_CREATE( bmath_mf3_s );
    bmath_mf3_s* m2 = BLM_CREATE( bmath_mf3_s );
    bmath_mf3_s* m3 = BLM_CREATE( bmath_mf3_s );

    bmath_mf3_s_set_size( m1, n, n );
    bmath_mf3_s_set_size( m2, n, n );
    bmath_mf3_s_set_size( m3, n, n );

    bmath_mf3_s_set_random( m1, true, true, 0, 1.0, -1.0, 1.0, NULL );

    // make m1 lower triangular
    for( sz_t i = 0; i < n; i++ ) for( sz_t j = i + 1; j < n; j++ ) m1->data[ i * m1->stride + j ] = 0;

    CPU_TIME_TO_STDOUT( fp_ltr_inv_htp( m1, m2 ) );
    bmath_mf3_s_mul_htp( m1, m2, m3 );

    ASSERT( bmath_mf3_s_is_near_one( m3, 1E-8 ) );

    BLM_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void mf3_eval( void )
{
    BCORE_LIFE_INIT();
    BCORE_LIFE_CREATE( bmath_mfx_eval_s, eval );
    BCORE_LIFE_CREATE( bmath_arr_mfx_eval_s, arr_eval );
    BCORE_LIFE_CREATE( bmath_arr_mfx_eval_s, arr_eval_sqr ); // square matrices

    eval->density = 1;
    eval->full  = false;
    eval->create_u_log = false;
    eval->create_v_log = false;
    eval->create_a_log = false;
//    st_s_push_sc( &eval->a_img_file, "/home/johannes/temp/a_img.pnm" );
//    st_s_push_sc( &eval->u_img_file, "/home/johannes/temp/u_img.pnm" );
//    st_s_push_sc( &eval->v_img_file, "/home/johannes/temp/v_img.pnm" );
    eval->test0 = true;
    eval->test1 = true;

    eval->assert_all = true;
    eval->near_limit_f2 = 1E-3;
    eval->near_limit_f3 = 1E-6;

//    test_ltr_inv_htp( 2048, ltr_inv_htp );
//    test_ltr_inv(     2048, ltr_inv );

//    eval->test1 = false;
//    eval->v_log = true;
//    eval->rows = 10; eval->cols = 10; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 1000; eval->cols = 1000; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 992; eval->cols = 992; bmath_arr_mfx_eval_s_push( arr_eval, eval );

//    eval->rows = 1341; eval->cols = 1341;  eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval_sqr, eval );
    eval->rows = 200; eval->cols = 200;  eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval_sqr, eval );

//    eval->rows = 233; eval->cols = 654; eval->dim3 = 739; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 200; eval->cols = 200; eval->dim3 = 200; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 654; eval->cols = 233; eval->dim3 = 739; bmath_arr_mfx_eval_s_push( arr_eval, eval );

    eval->rows = 1024; eval->cols = 1024; eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval, eval );

    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul    , bmath_mf3_s_mul_esp );
    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_mul    , bmath_mf2_s_mul_esp );

    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_htp_mul, bmath_mf3_s_htp_mul_esp );
    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_htp_mul, bmath_mf2_s_htp_mul_esp );

    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul    , bmath_mf3_s_norder_mul );
    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul    , bmath_mf3_s_morder_mul );
    //BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul    , bmath_mf3_s_zorder_mul );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_mul,         bmath_mf2_s_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_mul_htp,     bmath_mf2_s_mul_htp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_htp_mul,     bmath_mf2_s_htp_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_htp_mul_htp, bmath_mf2_s_htp_mul_htp );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul,         bmath_mf3_s_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul_htp,     bmath_mf3_s_mul_htp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_htp_mul,     bmath_mf3_s_htp_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_htp_mul_htp, bmath_mf3_s_htp_mul_htp );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_ubd, bmath_mf2_s_ubd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_lbd, bmath_mf2_s_lbd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_svd, bmath_mf2_s_svd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_ubd, bmath_mf3_s_ubd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_lbd, bmath_mf3_s_lbd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_svd, bmath_mf3_s_svd );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_qrd    , bmath_mf2_s_qrd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_qrd_pmt, bmath_mf2_s_qrd_pmt );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_qrd    , bmath_mf3_s_qrd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_qrd_pmt, bmath_mf3_s_qrd_pmt );
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_lqd    , bmath_mf2_s_lqd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_pmt_lqd, bmath_mf2_s_pmt_lqd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_lqd    , bmath_mf3_s_lqd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_pmt_lqd, bmath_mf3_s_pmt_lqd );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_trd    , bmath_mf2_s_trd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_evd_htp, bmath_mf2_s_evd_htp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_trd    , bmath_mf3_s_trd );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_evd_htp, bmath_mf3_s_evd_htp );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_cld    , bmath_mf2_s_decompose_cholesky );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_lud    , bmath_mf2_s_decompose_luc );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_inv    , bmath_mf2_s_inv );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_pdf_inv, bmath_mf2_s_pdf_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_piv    , bmath_mf2_s_piv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_hsm_piv, bmath_mf2_s_hsm_piv );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_cld    , decompose_cholesky );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_cld    , bmath_mf3_s_decompose_cholesky );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_lud    , bmath_mf3_s_decompose_luc );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_inv    , bmath_mf3_s_inv );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_pdf_inv, bmath_mf3_s_pdf_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_piv    , bmath_mf3_s_piv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_hsm_piv, bmath_mf3_s_hsm_piv );

//    st_s_print_d( bcore_spect_status() );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void mf3_s_eval( void )
{
    BCORE_LIFE_INIT();

    bmath_mf3_sx_s_convolution_eval1();
    bmath_mf3_sf_s_convolution_eval1();

    bmath_mf3_sx_s_mul_eval();
    bmath_mf3_sf_s_mul_eval();

    bmath_mf3_sx_s_mul_htp_eval();
    bmath_mf3_sf_s_mul_htp_eval();

    bmath_mf3_sx_s_htp_mul_eval();
    bmath_mf3_sf_s_htp_mul_eval();

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

int main( void )
{
    bcore_register_signal_handler( bmath_signal_handler );
    if( bcore_plant_run_globally() ) { bcore_down( true ); return 0; }

    //test_simd();
    //return 0;

    //bmath_quicktypes_to_stdout( NULL );
    //return 0;

    bmath_hwflags_to_stdout();

//    bmath_mf3_s_mul_add_cps_selftest();

//    bcore_run_signal_selftest( typeof( "bmath_smf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf3_sx" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf3_sf" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_vf2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_vf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_arr_vf2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_arr_vf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_cf2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_cf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_fourier_f2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_fourier_f3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_vcf2" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_vcf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_predictor" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mfx_eval" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_pmt" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_simd" ), NULL );
//    bmath_mf3_sx_s_htp_mul_eval();

    mf3_eval();
//    mf3_s_eval();

//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_hf3" ), NULL ) );
//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_adaptive_mlp" ), NULL ) );

    //snn_selftest2();
    bcore_down( true );
    return 0;
}
