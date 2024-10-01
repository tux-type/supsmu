#include <math.h>
#include <stddef.h>
#pragma once

typedef struct {
  float_t w;
  float_t b;
} Params;

Params fit_linear(float_t *, float_t *, size_t);
Params *fit_local_linear(float_t *, float_t *, size_t, size_t);