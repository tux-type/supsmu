#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void linear(const float_t *x, const float_t w, const float_t b, float_t *yhat,
            const size_t len) {
  for (uint32_t i = 0; i < len; i++) {
    yhat[i] = w * x[i] + b;
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

void grads(const float_t *x, const float_t *y, const float_t w, const float_t b,
           float_t *dj_dw, float_t *dj_db, const size_t len) {
  *dj_dw = 0;
  *dj_db = 0;

  float_t *yhat = malloc(sizeof(float_t) * len);
  if (yhat == NULL) {
    printf("Failed to allocate %zu length array for predictions\n", len);
    exit(EXIT_FAILURE);
  }
  linear(x, w, b, yhat, len);

  for (size_t i = 0; i < len; i++) {
    *dj_dw += (yhat[i] - y[i]) * x[i];
    *dj_db += (yhat[i] - y[i]);
  }
  *dj_dw = *dj_dw / len;
  *dj_db = *dj_db / len;

  free(yhat);
};

void update(float_t *w, float_t *b, const float_t dj_dw, const float_t dj_db,
            const float_t alpha) {
  *w = *w - alpha * dj_dw;
  *b = *b - alpha * dj_db;
}

void gradient_descent(const float *x, const float_t *y, float_t *w, float_t *b,
                      const float_t alpha, const size_t num_iters,
                      float_t (*cost_fn)(const float_t *y, const float_t *yhat,
                                         const size_t len),
                      void (*grad_fn)(const float_t *x, const float_t *y,
                                      const float_t w, const float_t b,
                                      float_t *dj_dw, float_t *dj_db,
                                      const size_t len),
                      const size_t len) {
  float_t dj_dw = 0;
  float_t dj_db = 0;
  for (size_t i = 0; i < num_iters; i++) {
    grad_fn(x, y, *w, *b, &dj_dw, &dj_db, len);
    grad_fn(x, y, *w, *b, &dj_dw, &dj_db, len);

    update(w, b, dj_dw, dj_db, alpha);
  }
}

void fit(float_t *x, float_t *y, float_t *w, float_t *b, size_t len) {
  *w = 0;
  *b = 0;
  float_t alpha = 0.01;
  size_t num_iters = 1000000;
  gradient_descent(x, y, w, b, alpha, num_iters, &l2, &grads, len);
}
