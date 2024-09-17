#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

void linear(const float_t *x, float_t *yhat, const float_t w, const float_t b,
            const size_t len) {
  for (uint32_t i = 0; i < len; i++) {
    yhat[i] = w * x[i] + b;
  }
}

float_t l2(float_t *yhat, float_t *y, size_t len) {
  float_t loss = 0;
  for (size_t i = 0; i < len; i++) {
    loss += pow((y[i] - yhat[i]), 2);
  }
  loss = 0.5 * len * loss;
  return loss;
}

void grads(float_t *dj_dw, float_t *dj_db, float_t *x, float_t *yhat,
           float_t *y, float_t w, float_t b, size_t len) {
  assert(*dj_dw == 0);
  assert(*dj_db == 0);

  for (size_t i = 0; i < len; i++) {
    *dj_dw += (yhat[i] - y[i]) * x[i];
    *dj_db += (yhat[i] - y[i]);
  }
  *dj_dw = *dj_dw / len;
  *dj_db = *dj_db / len;
};

void update(float_t *w, float_t *b, float_t dj_dw, float_t dj_db,
            float_t alpha) {
  *w = *w - alpha * dj_dw;
  *b = *b - alpha * dj_db;
}
