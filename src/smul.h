/// Author & Copyright (C) 2018 Johannes Bernhard Steffens. All rights reserved.

#ifndef SMUL_H
#define SMUL_H

#include "bmath_std.h"
#include "bmath_smf3.h"

void bmath_smf3_s_mul1( const bmath_smf3_s* o, const bmath_smf3_s* m, bmath_smf3_s* r );

void bmath_smf3_s_mul_eval();

void bmath_smf3_s_convolution_eval1();
void bmath_smf3_s_convolution_eval2();

#endif  // SMUL_H
