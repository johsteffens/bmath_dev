/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#include <stdio.h>
#include "bmath_std.h"
#include "cnn.h"

//----------------------------------------------------------------------------------------------------------------------

void cnn_selftest()
{
    BCORE_LIFE_INIT();

    BCORE_LIFE_CREATE( bmath_cnn_s, cnn );

    cnn->input_size = 32;
    cnn->input_step = 1;
    cnn->input_convolution_size = 4;
    cnn->input_kernels = 8;
    cnn->kernels_rate = 0;
    cnn->random_state = 124;

    cnn->act_mid = bmath_cnn_act_leaky_relu();
    cnn->act_out = bmath_cnn_act_tanh();

//    bcore_txt_ml_a_to_stdout( cnn );
//    bcore_bin_ml_a_from_file( cnn, "temp/cnn.bin" );

    bmath_cnn_s_setup( cnn );
    bmath_cnn_s_arc_to_sink( cnn, BCORE_STDOUT );

    /* Learn differentiating between a sine wave of arbitrary amplitude and frequency from
       a random walk curve.
    */
    BCORE_LIFE_CREATE_AUT( bcore_arr_sr_s, pos_set );
    BCORE_LIFE_CREATE_AUT( bcore_arr_sr_s, neg_set );

    sz_t samples = 10000;
    u2_t rval = 123;
    for( sz_t i = 0; i < samples; i++ )
    {
        bmath_vf3_s* pos_vec = bmath_vf3_s_create();
        bmath_vf3_s* neg_vec = bmath_vf3_s_create();
        bmath_vf3_s_set_size( pos_vec, cnn->input_size );
        bmath_vf3_s_set_size( neg_vec, cnn->input_size );

        f3_t omega = 1.0 * f3_pi() * f3_rnd_pos( &rval );
        f3_t amplitude = 4.0 * f3_rnd_pos( &rval );

        f3_t rwalker = f3_rnd_sym( &rval );

        for( sz_t i = 0; i < cnn->input_size; i++ )
        {
            rwalker += f3_rnd_sym( &rval );
            f3_t vp = sin( omega * i ) * amplitude;
            f3_t vr = rwalker;
            f3_t f = ( ( f3_t )i ) / cnn->input_size;
            f3_t vn = vp * f + vr * ( 1.0 - f );

            pos_vec->data[ i ] = vp;
            neg_vec->data[ i ] = vn;
        }

        bcore_arr_sr_s_push_sr( pos_set, sr_asd( pos_vec ) );
        bcore_arr_sr_s_push_sr( neg_set, sr_asd( neg_vec ) );
    }

    sz_t epochs = 100;
    f3_t learn_step = 0.0001;
    f3_t pos_tgt = 0.9;
    f3_t neg_tgt = -pos_tgt;

    for( sz_t i = 0; i < epochs; i++ )
    {
        f3_t err = 0;
        for( sz_t j = 0; j < samples; j++ )
        {
            const bmath_vf3_s* pos_vec = pos_set->data[ j ].o;
            const bmath_vf3_s* neg_vec = neg_set->data[ j ].o;

            f3_t pos_est = bmath_cnn_s_learn_1( cnn, pos_vec, pos_tgt, learn_step );
            f3_t neg_est = bmath_cnn_s_learn_1( cnn, neg_vec, neg_tgt, learn_step );

            err += f3_sqr( pos_est - pos_tgt );
            err += f3_sqr( neg_est - neg_tgt );
        }

        err = f3_srt( err / ( samples * 2 ) );

        bcore_msg_fa( "#pl6 {#<sz_t>}: err = #<f3_t>\n", i, err );
    }

    // bcore_bin_ml_a_to_file( cnn, "temp/cnn.bin" );

    BCORE_LIFE_RETURN();
}

//----------------------------------------------------------------------------------------------------------------------

