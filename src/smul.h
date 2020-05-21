/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#ifndef SMUL_H
#define SMUL_H

#include "bmath_std.h"
#include "bmath_xsmf3.h"
#include "bmath_asmf3.h"

void bmath_xsmf3_s_mul_eval();
void bmath_xsmf3_s_mul_htp_eval();
void bmath_xsmf3_s_htp_mul_eval();

void bmath_asmf3_s_mul_eval();
void bmath_asmf3_s_mul_htp_eval();
void bmath_asmf3_s_htp_mul_eval();

void bmath_xsmf3_s_convolution_eval1();
void bmath_asmf3_s_convolution_eval1();

void bmath_xsmf3_s_convolution_eval2();

#endif  // SMUL_H
