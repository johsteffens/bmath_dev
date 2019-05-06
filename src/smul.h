/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#ifndef SMUL_H
#define SMUL_H

#include "bmath_std.h"
#include "bmath_mf3_sx.h"
#include "bmath_mf3_sf.h"

void bmath_mf3_sx_s_mul_eval();
void bmath_mf3_sx_s_mul_htp_eval();
void bmath_mf3_sx_s_htp_mul_eval();

void bmath_mf3_sf_s_mul_eval();
void bmath_mf3_sf_s_mul_htp_eval();
void bmath_mf3_sf_s_htp_mul_eval();

void bmath_mf3_sx_s_convolution_eval1();
void bmath_mf3_sf_s_convolution_eval1();

void bmath_mf3_sx_s_convolution_eval2();

#endif  // SMUL_H
