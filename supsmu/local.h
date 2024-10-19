#include <math.h>
#include <stddef.h>
#pragma once

typedef struct {
  float_t w;
  float_t b;
} Params;

Params fit_linear(float_t *, float_t *, size_t);
void fit_local_linear(Params *, float_t *, float_t *, size_t, size_t);
void local_linear_forward(float_t *, Params *, float_t *, size_t);
