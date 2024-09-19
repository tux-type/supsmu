#include "linear.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void linear(const float_t *x, const Params *params, float_t *yhat,
            const size_t len) {
  for (uint32_t i = 0; i < len; i++) {
    yhat[i] = params->w * x[i] + params->b;
  }
}

float_t l2(const float_t *y, const float_t *yhat, const size_t len) {
  float_t loss = 0;
  for (size_t i = 0; i < len; i++) {
    loss += pow((y[i] - yhat[i]), 2.0);
  }
  loss = (1.0 / (2.0 * len)) * loss;
  return loss;
}

void calc_grads(const float_t *x, const float_t *y, const Params *params,
                Grads *grads, const size_t len) {
  grads->dj_dw = 0;
  grads->dj_db = 0;

  float_t *yhat = malloc(sizeof(float_t) * len);
  if (yhat == NULL) {
    printf("Failed to allocate %zu length array for predictions\n", len);
    exit(EXIT_FAILURE);
  }
  linear(x, params, yhat, len);

  for (size_t i = 0; i < len; i++) {
    grads->dj_dw += (yhat[i] - y[i]) * x[i];
    grads->dj_db += (yhat[i] - y[i]);
  }
  grads->dj_dw = grads->dj_dw / len;
  grads->dj_db = grads->dj_db / len;

  free(yhat);
};

void update(Params *params, Grads *grads, const float_t alpha) {
  params->w = params->w - alpha * grads->dj_dw;
  params->b = params->b - alpha * grads->dj_db;
}

Params gradient_descent(
    const float *x, const float_t *y, const float_t alpha,
    const size_t num_iters,
    float_t (*cost_fn)(const float_t *y, const float_t *yhat, const size_t len),
    void (*grad_fn)(const float_t *x, const float_t *y, const Params *params,
                    Grads *grads, const size_t len),
    const size_t len) {
  float_t dj_dw = 0;
  float_t dj_db = 0;
  Params params = {.w = 0, .b = 0};
  Grads grads = {.dj_dw = 0, .dj_db = 0};

  for (size_t i = 0; i < num_iters; i++) {
    grad_fn(x, y, &params, &grads, len);

    update(&params, &grads, alpha);
  }
  return params;
}

Params fit(float_t *x, float_t *y, size_t len) {
  float_t alpha = 0.01;
  size_t num_iters = 1000000;
  Params params =
      gradient_descent(x, y, alpha, num_iters, &l2, &calc_grads, len);
  return params;
}
