#include "local.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void check_alloc(void *p, size_t len) {
  if (p == NULL) {
    printf("Failed to allocate %zu length array\n", len);
    exit(EXIT_FAILURE);
  }
}

float_t *supsmu(float_t *x, float_t *y, size_t len) {
  float_t spans[] = {0.05, 0.2, 0.5};

  // TODO: TIDY UP AND FIGURE OUT CONVENIENT FREE
  Params **local_params = malloc(sizeof(Params *) * 3);
  check_alloc(local_params, sizeof(Params *) * 3);
  float_t **yhats = malloc(sizeof(float_t *) * 3);
  check_alloc(yhats, sizeof(float_t *) * 3);
  float_t **residuals = malloc(sizeof(float_t *) * 3);
  check_alloc(residuals, sizeof(float_t *) * 3);

  Params **local_residual_params = malloc(sizeof(Params *) * 3);
  check_alloc(local_residual_params, sizeof(Params *) * 3);
  float_t **residuals_yhats = malloc(sizeof(float_t *) * 3);
  check_alloc(residuals_yhats, sizeof(float_t *) * 3);


  for (size_t i = 0; i < 3; i++) {
    // Mallocs
    local_params[i] = malloc(sizeof(Params) * len);
    check_alloc(local_params, sizeof(Params) * len);
    yhats[i] = malloc(sizeof(float_t) * len);
    check_alloc(yhats, sizeof(float_t) * len);
    residuals[i] = malloc(sizeof(float_t) * len);
    check_alloc(residuals, sizeof(float_t) * len);

    // Calc. residuals for each span
    fit_local_linear(local_params[i], x, y, len, spans[i] * len);
    forward_local_linear(yhats[i], local_params[i], x, len);
    for (size_t j = 0; j < len; j++) {
      residuals[i][j] = y[j] - yhats[i][j];
    }

    local_residual_params[i] = malloc(sizeof(Params) * len);
    check_alloc(local_params, sizeof(Params) * len);
    residuals_yhats[i] = malloc(sizeof(float_t) * len);
    check_alloc(residuals_yhats, sizeof(float_t) * len);
    
    // Smooth residuals using 0.5
    fit_local_linear(local_residual_params[i], x, y, len, 0.5 * len);
    forward_local_linear(residuals_yhats[i], local_params[i], x, len);
  }


  Params* smoothing_params = malloc(sizeof(Params) * len);
  float_t* smoothed_yhat = malloc(sizeof(float_t) * len);
  for (size_t j = 0; j < len; j++) {
    // Single point residual for the 3 spans 0.05, 0.2, 0.5
    float_t residuals_j[] = { residuals_yhats[0][j], residuals_yhats[1][j], residuals_yhats[2][j] };

    size_t smallest_k = 0;
    for (size_t k = 0; k < 3; k++) {
      if (residuals_j[k] < residuals_j[smallest_k]) {
        size_t smallest_k = k;
      }
    }

    // Select span with lowest residual for point j
    smoothing_params[j] = local_params[smallest_k][j];
  }

  forward_local_linear(smoothed_yhat, smoothing_params, x, len);
  return smoothed_yhat;
}
