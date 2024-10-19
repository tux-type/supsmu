#include <math.h>
#include <stddef.h>
#pragma once

typedef struct {
  float_t w;
  float_t b;
} Params;

Params fit_linear(const float_t *, const float_t *, const size_t);
void fit_local_linear(Params *, const float_t *, const float_t *, const size_t, const size_t);
void forward_local_linear(float_t *, const Params *, const float_t *, const size_t);
