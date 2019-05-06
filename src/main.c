/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <sys/time.h>

#include "bmath_std.h"
#include "bmath_mf3_eval.h"

#include "svd.h"
#include "mul.h"
#include "smul.h"
#include "qrd.h"
#include "cnn.h"
#include "snn.h"
#include "zorder.h"

void mf3_eval( void )
{
    BCORE_LIFE_INIT();
    BCORE_LIFE_CREATE( bmath_mf3_eval_s, eval );
    BCORE_LIFE_CREATE( bmath_arr_mf3_eval_s, arr_eval );

    eval->density = 1;
    eval->full  = false;
    eval->create_u_log = false;
    eval->create_v_log = false;
    eval->create_a_log = false;
    st_s_push_sc( &eval->a_img_file, "/home/johannes/temp/a_img.pnm" );
    st_s_push_sc( &eval->u_img_file, "/home/johannes/temp/u_img.pnm" );
//    st_s_push_sc( &eval->v_img_file, "/home/johannes/temp/v_img.pnm" );
    eval->test0 = true;
    eval->test1 = true;

    eval->assert_all = true;
    eval->near_limit = 1E-6;

//    eval->test1 = false;
//    eval->v_log = true;
//    eval->rows = 10; eval->cols = 10; bmath_arr_mf3_eval_s_push( arr_eval, eval );
//    eval->rows = 1000; eval->cols = 1000; bmath_arr_mf3_eval_s_push( arr_eval, eval );
//    eval->rows = 992; eval->cols = 992; bmath_arr_mf3_eval_s_push( arr_eval, eval );

    eval->rows = 1000; eval->cols = 1000;

    bmath_arr_mf3_eval_s_push( arr_eval, eval );

//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_norder_mul );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_morder_mul );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_zorder_mul );

//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul,     ( fp_t )bmath_mf3_s_mul );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul_htp, ( fp_t )bmath_mf3_s_mul_htp );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_htp_mul, ( fp_t )bmath_mf3_s_htp_mul );

//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_qrd    , ( fp_t )bmath_mf3_s_qrd );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_qrd    , ( fp_t )bmath_mf3_s_qrd2 );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_ua     , ( fp_t )bmath_mf3_s_qrd2 );
//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_ua     , ( fp_t )bmath_mf3_s_qrd3 );

//    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_ubd,     ( fp_t )bmath_mf3_s_ubd );

/*
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_qrd    , ( fp_t )bmath_mf3_s_qrd );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_mul_esp );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul_htp, ( fp_t )bmath_mf3_s_mul_htp_esp  );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_qrd_pmt, ( fp_t )bmath_mf3_s_qrd_pmt );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_ubd,     ( fp_t )bmath_mf3_s_ubd );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_lbd,     ( fp_t )bmath_mf3_s_lbd );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_cld    , ( fp_t )bmath_mf3_s_decompose_cholesky );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_lud    , ( fp_t )bmath_mf3_s_decompose_luc );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_lqd    , ( fp_t )bmath_mf3_s_lqd );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_pmt_lqd, ( fp_t )bmath_mf3_s_pmt_lqd );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_inv    , ( fp_t )bmath_mf3_s_inv );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_pdf_inv, ( fp_t )bmath_mf3_s_pdf_inv );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_piv    , ( fp_t )bmath_mf3_s_piv );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_hsm_piv, ( fp_t )bmath_mf3_s_hsm_piv );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_svd    , ( fp_t )bmath_mf3_s_svd );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_trd    , ( fp_t )bmath_mf3_s_trd );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_evd_htp, ( fp_t )bmath_mf3_s_evd_htp );
*/
//    st_s_print_d( bcore_spect_status() );

    BCORE_LIFE_DOWN();
}

void mf3_s_eval( void )
{
    BCORE_LIFE_INIT();

//    bmath_mf3_sx_s_convolution_eval1();
//    bmath_mf3_sf_s_convolution_eval1();

//    bmath_mf3_sx_s_mul_eval();
    bmath_mf3_sf_s_mul_eval();

//    bmath_mf3_sx_s_mul_htp_eval();
    bmath_mf3_sf_s_mul_htp_eval();

//    bmath_mf3_sx_s_htp_mul_eval();
    bmath_mf3_sf_s_htp_mul_eval();

    BCORE_LIFE_DOWN();
}

int main( void )
{
    bcore_register_signal_handler( bmath_signal_handler );
    if( bcore_precoder_run_globally() ) { bcore_down( true ); return 0; }

//    bmath_quicktypes_to_stdout( NULL );
//    return 0;

    bmath_hwflags_to_stdout();
//    bcore_run_signal_selftest( typeof( "bmath_smf3" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_mf3_sx" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_mf3_sf" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf3" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_vector" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_predictor" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_mf3_eval" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_pmt" ), NULL );
//    bcore_run_signal_selftest( typeof( "bmath_simd" ), NULL );
//    bmath_mf3_sx_s_htp_mul_eval();

    // mf3_eval();
    mf3_s_eval();

//    CPU_TIME_TO_STDOUT( cnn_1d_selftest2() );

//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_adaptive_cnn_1d" ), NULL ) );
//    CPU_TIME_TO_STDOUT( bcore_run_signal_selftest( typeof( "bmath_adaptive_mlp" ), NULL ) );

    //snn_selftest2();
    bcore_down( true );
    return 0;
}
