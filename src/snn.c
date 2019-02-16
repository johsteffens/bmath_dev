/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "snn.h"
#include "bmath_spect_adaptive.h"

//----------------------------------------------------------------------------------------------------------------------

void snn_selftest1()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_snn_s, snn );

    snn->input_size = 32;
    snn->input_kernels = 16;
    snn->layers = 8;
    snn->kernels_rate = 0;
    snn->random_state = 124;

    snn->act_mid = bmath_snn_act_leaky_relu();
    snn->act_out = bmath_snn_act_tanh();

    bmath_snn_s_setup( snn, true );
    bmath_snn_s_arc_to_sink( snn, BCORE_STDOUT );

    /* Learn differentiating between a sine wave of arbitrary amplitude and frequency from
       a random walk curve.
    */
    BCORE_LIFE_CREATE( bcore_arr_sr_s, pos_set_trn );
    BCORE_LIFE_CREATE( bcore_arr_sr_s, neg_set_trn );
    BCORE_LIFE_CREATE( bcore_arr_sr_s, pos_set_tst );
    BCORE_LIFE_CREATE( bcore_arr_sr_s, neg_set_tst );

    sz_t samples = 10000;
    u2_t rval = 123;
    for( sz_t i = 0; i < samples * 2; i++ )
    {
        bmath_vf3_s* pos_vec = bmath_vf3_s_create();
        bmath_vf3_s* neg_vec = bmath_vf3_s_create();
        bmath_vf3_s_set_size( pos_vec, snn->input_size );
        bmath_vf3_s_set_size( neg_vec, snn->input_size );

        f3_t omega = 1.0 * f3_pi() * f3_rnd_pos( &rval );
        f3_t amplitude = 4.0 * f3_rnd_pos( &rval );

        f3_t rwalker = f3_rnd_sym( &rval );

        for( sz_t i = 0; i < snn->input_size; i++ )
        {
            rwalker += f3_rnd_sym( &rval );
            f3_t vp = sin( omega * i ) * amplitude;
            f3_t vr = rwalker;
            f3_t vn = vr;

            pos_vec->data[ i ] = vp;
            neg_vec->data[ i ] = vn;
        }

        if( ( i & 1 ) == 0 )
        {
            bcore_arr_sr_s_push_sr( pos_set_trn, sr_asd( pos_vec ) );
            bcore_arr_sr_s_push_sr( neg_set_trn, sr_asd( neg_vec ) );
        }
        else
        {
            bcore_arr_sr_s_push_sr( pos_set_tst, sr_asd( pos_vec ) );
            bcore_arr_sr_s_push_sr( neg_set_tst, sr_asd( neg_vec ) );
        }
    }

    sz_t epochs = 30;
    snn->adapt_step = 0.0001;
    snn->decay_step = 0.0001 * snn->adapt_step;
    f3_t pos_tgt = 0.9;
    f3_t neg_tgt = -pos_tgt;

    for( sz_t i = 0; i < epochs; i++ )
    {
        f3_t err = 0;
        for( sz_t j = 0; j < samples; j++ )
        {
            const bmath_vf3_s* pos_vec = pos_set_trn->data[ j ].o;
            const bmath_vf3_s* neg_vec = neg_set_trn->data[ j ].o;
            f3_t pos_est = bmath_snn_s_adapt_1( snn, pos_vec, pos_tgt );
            f3_t neg_est = bmath_snn_s_adapt_1( snn, neg_vec, neg_tgt );
            err += f3_sqr( pos_est - pos_tgt );
            err += f3_sqr( neg_est - neg_tgt );
        }

        err = f3_srt( err / ( samples * 2 ) );

        bcore_msg_fa( "#pl6 {#<sz_t>}: err = #<f3_t>\n", i, err );
    }

    bcore_bin_ml_a_to_file( snn, "temp/snn.bin" );
    BCORE_LIFE_CREATE( bmath_snn_s, snn_tst );
    bcore_bin_ml_a_from_file( snn_tst, "temp/snn.bin" );

//    bcore_txt_ml_a_to_stdout( snn );
//    bcore_txt_ml_a_to_stdout( snn_tst );

    {
        f3_t err = 0;
        for( sz_t j = 0; j < samples; j++ )
        {
            const bmath_vf3_s* pos_vec = pos_set_tst->data[ j ].o;
            const bmath_vf3_s* neg_vec = neg_set_tst->data[ j ].o;
            f3_t pos_est = bmath_snn_s_query_1( snn_tst, pos_vec );
            f3_t neg_est = bmath_snn_s_query_1( snn_tst, neg_vec );
            err += f3_sqr( pos_est - pos_tgt );
            err += f3_sqr( neg_est - neg_tgt );
        }

        err = f3_srt( err / ( samples * 2 ) );
        bcore_msg_fa( "tst_err = #<f3_t>\n", err );
    }
    BCORE_LIFE_RETURN();
}

//----------------------------------------------------------------------------------------------------------------------

void snn_selftest2()
{
    BCORE_LIFE_INIT();
    BCORE_LIFE_CREATE( bmath_snn_s, snn );

    snn->input_kernels = 16;
    snn->layers = 8;
    snn->kernels_rate = 0;
    snn->random_state = 124;
    snn->act_mid = bmath_snn_act_leaky_relu();
    snn->act_out = bmath_snn_act_tanh();

    bmath_adaptive_a_test( ( bmath_adaptive* )snn );

    BCORE_LIFE_RETURN();
}

//----------------------------------------------------------------------------------------------------------------------

