#include "local.h"
#include <assert.h>
#include <stdio.h>

Params fit_linear(const float_t *x, const float_t *y, const size_t len) {
  Params params = {.w = 0, .b = 0};

  if (len == 1) {
    params.w = 0;
    params.b = y[0];
    return params;
  }

  float_t sum_x = 0;
  float_t sum_y = 0;
  float_t sum_x2 = 0;
  float_t sum_xy = 0;
  for (size_t i = 0; i < len; i++) {
    sum_x += x[i];
    sum_y += y[i];
    sum_x2 += (x[i] * x[i]);
    sum_xy += (x[i] * y[i]);
  }

  float_t denominator = ((len * sum_x2) - (sum_x * sum_x));

  // Return constant model if denominator is 0 (all x2 values add up to 0)
  if (denominator == 0) {
    params.w = 0;
    params.b = sum_y / len;
    return params;
  }

  params.w = ((len * sum_xy) - (sum_x * sum_y)) / denominator;
  params.b = ((sum_y * sum_x2) - (sum_x * sum_xy)) / denominator;
  if (params.w > 1000) {
    printf("len: %lu\n", len);
    for (size_t i = 0; i < len; i++) {
      printf("y[i]: %lf\n", y[i]);
    }
    printf("sum_x: %lf\n", sum_x);
    printf("sum_y: %lf\n", sum_y);
    printf("sum_x2: %lf\n", sum_x2);
    printf("sum_xy: %lf\n", sum_xy);
    printf("params.w: %lf\n", params.w);
    printf("params.b: %lf\n", params.b);
  }
  return params;
}

void fit_local_linear(Params *local_params, const float_t *x, const float_t *y,
                      const size_t len, size_t J) {
  size_t half_J = J / 2;
  for (size_t i = 0; i < len; i++) {
    // Avoid out of bounds where J is smaller/larger than remaining array
    size_t start = (i < half_J) ? 0 : i - half_J;
    size_t end = (i + half_J) >= len ? len - 1 : i + half_J;

    size_t actual_J = end - start + 1;

    assert((start + actual_J) <= len);

    local_params[i] = fit_linear(&x[start], &y[start], actual_J);
  }
}

void forward_local_linear(float_t *yhat, const Params *local_params,
                          const float_t *x, const size_t len) {
  for (size_t i = 0; i < len; i++) {
    yhat[i] = x[i] * local_params[i].w + local_params[i].b;
  }
}
