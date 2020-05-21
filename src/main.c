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

//    eval->test1 = false;
//    eval->v_log = true;
//    eval->rows = 10; eval->cols = 10; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 1000; eval->cols = 1000; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 992; eval->cols = 992; bmath_arr_mfx_eval_s_push( arr_eval, eval );

    eval->rows = 1024; eval->cols = 1024;  eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval_sqr, eval );
//    eval->rows = 200; eval->cols = 200;  eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval_sqr, eval );

//    eval->rows = 654; eval->cols = 233; eval->dim3 = 739; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 233; eval->cols = 654; eval->dim3 = 739; bmath_arr_mfx_eval_s_push( arr_eval, eval );
//    eval->rows = 200; eval->cols = 200; eval->dim3 = 200; bmath_arr_mfx_eval_s_push( arr_eval, eval );
    eval->rows = 1024; eval->cols = 1024; eval->dim3 = -1; bmath_arr_mfx_eval_s_push( arr_eval, eval );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_mul    , bmath_mf3_s_mul_esp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_mul    , bmath_mf2_s_mul_esp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf3_s_htp_mul, bmath_mf3_s_htp_mul_esp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, mf2_s_htp_mul, bmath_mf2_s_htp_mul_esp );

//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf2_s_mul,         bmath_xsmf2_s_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf3_s_mul,         bmath_xsmf3_s_mul );
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf2_s_mul_htp,     bmath_xsmf2_s_mul_htp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf3_s_mul_htp,     bmath_xsmf3_s_mul_htp );
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf2_s_htp_mul,     bmath_xsmf2_s_htp_mul );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf3_s_htp_mul,     bmath_xsmf3_s_htp_mul );
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf2_s_htp_mul_htp, bmath_xsmf2_s_htp_mul_htp );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, xsmf3_s_htp_mul_htp, bmath_xsmf3_s_htp_mul_htp );

    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf2_s_mul,         bmath_asmf2_s_mul );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf3_s_mul,         bmath_asmf3_s_mul );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf2_s_mul_htp,     bmath_asmf2_s_mul_htp );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf3_s_mul_htp,     bmath_asmf3_s_mul_htp );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf2_s_htp_mul,     bmath_asmf2_s_htp_mul );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf3_s_htp_mul,     bmath_asmf3_s_htp_mul );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf2_s_htp_mul_htp, bmath_asmf2_s_htp_mul_htp );
    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval, asmf3_s_htp_mul_htp, bmath_asmf3_s_htp_mul_htp );

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
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_cld    , bmath_mf2_s_decompose_cholesky );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_lud    , bmath_mf2_s_decompose_luc );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_inv    , bmath_mf2_s_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_pdf_inv, bmath_mf2_s_pdf_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_piv    , bmath_mf2_s_piv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf2_s_hsm_piv, bmath_mf2_s_hsm_piv );
//
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_cld    , bmath_mf3_s_decompose_cholesky );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_lud    , bmath_mf3_s_decompose_luc );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_inv    , bmath_mf3_s_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_pdf_inv, bmath_mf3_s_pdf_inv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_piv    , bmath_mf3_s_piv );
//    BMATH_ARR_MFX_EVAL_S_RUN_TO_STDOUT( arr_eval_sqr, mf3_s_hsm_piv, bmath_mf3_s_hsm_piv );

//    st_s_print_d( bcore_spect_status() );

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

void mf3_s_eval( void )
{
    BCORE_LIFE_INIT();

    bmath_xsmf3_s_convolution_eval1();
    bmath_asmf3_s_convolution_eval1();

    bmath_xsmf3_s_mul_eval();
    bmath_asmf3_s_mul_eval();

    bmath_xsmf3_s_mul_htp_eval();
    bmath_asmf3_s_mul_htp_eval();

    bmath_xsmf3_s_htp_mul_eval();
    bmath_asmf3_s_htp_mul_eval();

    BCORE_LIFE_DOWN();
}

//----------------------------------------------------------------------------------------------------------------------

int main( void )
{
    bcore_register_signal_handler( bmath_signal_handler, 0 );
    if( bcore_plant_run_globally() ) { bcore_down( true ); return 0; }

    //test_simd();
    //return 0;

    //bmath_quicktypes_to_stdout( NULL );
    //return 0;

    bmath_hwflags_to_stdout();

//    bmath_mf3_s_mul_add_cps_selftest();

//    bcore_run_signal_selftest( typeof( "bmath_smf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_xsmf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_asmf3" ), NULL );
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
//    bmath_xsmf3_s_htp_mul_eval();

    mf3_eval();
//    mf3_s_eval();

//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_hf3" ), NULL ) );
//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_adaptive_mlp" ), NULL ) );

    //snn_selftest2();
    bcore_down( true );
    return 0;
}
