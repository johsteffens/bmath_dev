/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <sys/time.h>

#include "bmath_std.h"
#include "bmath_mf3_eval.h"

#include "svd.h"
#include "mul.h"

void randomizer( void )
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_mf3_s, m1 );
    BCORE_LIFE_CREATE( bmath_mf3_s, a );

    uz_t m = 100;
    uz_t n = 100;
    uz_t rd = uz_min( m, n ) >> 1;

    bmath_mf3_s_set_size( m1, m, n );

    //void bmath_mf3_s_set_random( bmath_mf3_s* o, bl_t hsm, bl_t pdf, uz_t rd, f3_t density, f3_t min, f3_t max, u2_t* p_rval );

    bmath_mf3_s_set_random( m1, false, false, rd, 0.1, -1, 1, NULL );
    bmath_mf3_s_to_stdout( m1 );

    bmath_mf3_s_set_size( a, m, n );
    bmath_mf3_s_cpy(   m1, a );
    bmath_mf3_s_qrd( NULL, a );
//    bmath_mf3_s_to_stdout( a );

    if( rd > 0 )
    {
        uz_t rank = uz_min( m, n ) - rd;
        bcore_msg_fa( "#<f3_t>\n", a->data[ rank * ( a->stride + 1 ) ] / a->data[ 0 ] );
    }

    bmath_mf3_s_cpy(   m1, a );
    bmath_mf3_s_qrd_pmt( NULL, a, NULL );
//    bmath_mf3_s_to_stdout( a );

    if( rd > 0 )
    {
        uz_t rank = uz_min( m, n ) - rd;
        bcore_msg_fa( "#<f3_t>\n", a->data[ rank * ( a->stride + 1 ) ] / a->data[ 0 ] );
    }

    bcore_msg_fa( "#<f3_t>\n", f3_lim_eps );

    BCORE_LIFE_DOWN();
}

int main( void )
{
    bcore_register_signal_handler( bmath_signal_handler );

//    randomizer();
//    return 0;

//    bmath_quicktypes_to_stdout( NULL );
//    return 0;

    bcore_run_signal_selftest( typeof( "bmath_mf3" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_vector" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_predictor" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_mf3_eval" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_pmt" ), NULL );
    bcore_run_signal_selftest( typeof( "bmath_simd" ), NULL );

    BCORE_LIFE_INIT();
    BCORE_LIFE_CREATE( bmath_mf3_eval_s, eval );
    BCORE_LIFE_CREATE( bmath_arr_mf3_eval_s, arr_eval );

    bmath_hwflags_to_stdout();

    eval->density = 1;
    eval->full  = false;
    eval->log_u = false;
    eval->log_v = false;
    eval->log_a = false;
    eval->test1 = true;

    eval->assert_all = true;
    eval->near_limit = 1E-6;

//    eval->test1 = false;
//    eval->log_v = true;
//    eval->rows = 10; eval->cols = 10; bmath_arr_mf3_eval_s_push( arr_eval, eval );
//    eval->rows = 1000; eval->cols = 1000; bmath_arr_mf3_eval_s_push( arr_eval, eval );
//    eval->rows = 992; eval->cols = 992; bmath_arr_mf3_eval_s_push( arr_eval, eval );
    eval->rows = 1000; eval->cols = 1000; bmath_arr_mf3_eval_s_push( arr_eval, eval );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_mul2 );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_mul );
/*

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_mul_esp );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul_htp, ( fp_t )bmath_mf3_s_mul_htp );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul_htp, ( fp_t )bmath_mf3_s_mul_htp_esp  );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul_htp, ( fp_t )bmath_mf3_s_mul_htp_esp );
    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_mul    , ( fp_t )bmath_mf3_s_mul );

    bmath_arr_mf3_eval_s_run_to_stdout( arr_eval, TYPEOF_bmath_fp_mf3_s_qrd    , ( fp_t )bmath_mf3_s_qrd );
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

    bcore_down( true );
    return 0;
}
