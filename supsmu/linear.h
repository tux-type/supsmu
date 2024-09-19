#include <math.h>
#include <stddef.h>
#pragma once

typedef struct {
  float_t w;
  float_t b;
} Params;

typedef struct {
  float_t dj_dw;
  float_t dj_db;
} Grads;

Params fit(float_t *, float_t *, size_t);
