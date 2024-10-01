#include "linear.h"
#include <stdlib.h>

Params fit_linear(float_t *x, float_t *y, size_t len) {
  float_t sum_x = 0;
  float_t sum_y = 0;
  float_t sum_x2 = 0;
  float_t sum_xy = 0;
  for (size_t i = 0; i < len; i++) {
    sum_x += x[i];
    sum_y += y[i];
    sum_x2 += x[i] * x[i];
    sum_xy += x[i] * y[i];
  }

  Params params = {.w = 0, .b = 0};
  params.w =
      ((len * sum_xy) - (sum_x * sum_y)) / ((len * sum_x2) - (sum_x * sum_x));
  params.b = ((sum_y * sum_x2) - (sum_x * sum_xy)) /
             ((len * sum_x2) - (sum_x * sum_x));
  return params;
}

Params *fit_local_linear(float_t *x, float_t *y, size_t len, size_t J) {
  Params *local_params = malloc(sizeof(Params) * len);

  for (size_t i = 0; i < len; i++) {
    local_params[i] = fit_linear(&x[i], &y[i], J);
  }
  return local_params; // Caller expected to free.
}
