#include "local.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void check_alloc2(void *p, size_t len) {
  if (p == NULL) {
    printf("Failed to allocate %zu length array\n", len);
    exit(EXIT_FAILURE);
  }
}

float_t adj_factor(float_t *x, size_t idx, size_t start, size_t J) {
  assert(J > 0);
  float_t sum_x = 0;
  for (size_t i = start; i < start + J; i++) {
    sum_x += x[i];
  }
  float_t xbar_J = sum_x / (float_t)J;

  float_t v_J = 0;
  for (size_t i = start; i < start + J; i++) {
    v_J += pow(x[i] - xbar_J, 2);
  }

  float_t adj_factor = (1 - (1 / (float_t)J) - (pow(x[idx] - xbar_J, 2) / v_J));
  return adj_factor;
}

void write_csv(char *file_name, float_t **data, char *col_names,
               size_t num_cols, size_t num_rows, bool row_major) {
  FILE *fp = fopen(file_name, "w");
  if (fp == NULL) {
    printf("File %s could not be opened for writing.", file_name);
    exit(EXIT_FAILURE);
  }
  if (row_major) {
    fprintf(fp, "%s\n", col_names);
    for (size_t i = 0; i < num_rows; i++) {
      for (size_t j = 0; j < (num_cols - 1); j++) {
        fprintf(fp, "%lf,", data[i][j]);
      }
      fprintf(fp, "%lf", data[i][num_cols - 1]);
      fprintf(fp, "\n");
    }
  } else {
    fprintf(fp, "%s\n", col_names);
    for (size_t i = 0; i < num_rows; i++) {
      for (size_t j = 0; j < (num_cols - 1); j++) {
        fprintf(fp, "%lf,", data[j][i]);
      }
      fprintf(fp, "%lf", data[num_cols - 1][i]);
      fprintf(fp, "\n");
    }
  }
}

float_t *supsmu(float_t *x, float_t *y, size_t len) {
  float_t spans[] = {0.05, 0.2, 0.5};

  // TODO: TIDY UP AND FIGURE OUT CONVENIENT FREE
  Params **local_params = malloc(sizeof(Params *) * 3);
  check_alloc2(local_params, sizeof(Params *) * 3);
  float_t **yhats = malloc(sizeof(float_t *) * 3);
  check_alloc2(yhats, sizeof(float_t *) * 3);
  float_t **residuals = malloc(sizeof(float_t *) * 3);
  check_alloc2(residuals, sizeof(float_t *) * 3);

  Params **local_residual_params = malloc(sizeof(Params *) * 3);
  check_alloc2(local_residual_params, sizeof(Params *) * 3);
  float_t **residuals_yhats = malloc(sizeof(float_t *) * 3);
  check_alloc2(residuals_yhats, sizeof(float_t *) * 3);

  for (size_t j = 0; j < 3; j++) {
    size_t J = ceil(spans[j] * len);

    // Mallocs
    local_params[j] = malloc(sizeof(Params) * len);
    check_alloc2(local_params, sizeof(Params) * len);
    yhats[j] = malloc(sizeof(float_t) * len);
    check_alloc2(yhats, sizeof(float_t) * len);
    residuals[j] = malloc(sizeof(float_t) * len);
    check_alloc2(residuals, sizeof(float_t) * len);

    // Calc. residuals for each span
    // Using ceil since otherwise can end up with 0 size J span
    fit_local_linear(local_params[j], x, y, len, J);

    forward_local_linear(yhats[j], local_params[j], x, len);

    for (size_t i = 0; i < len; i++) {
      // TODO: Refactor as already performing this in local.c
      size_t half_J = J / 2;
      size_t start = (i < half_J) ? 0 : i - half_J;
      size_t end = (i + half_J) >= len ? len - 1 : i + half_J;

      size_t actual_J = end - start + 1;

      float_t adj_factor_i = adj_factor(x, i, start, actual_J);

      residuals[j][i] = (y[i] - yhats[j][i]) / adj_factor_i;
    }

    local_residual_params[j] = malloc(sizeof(Params) * len);
    check_alloc2(local_params, sizeof(Params) * len);
    residuals_yhats[j] = malloc(sizeof(float_t) * len);
    check_alloc2(residuals_yhats, sizeof(float_t) * len);

    // Smooth residuals using 0.5 - should it be 0.2 OR 0.5???
    fit_local_linear(local_residual_params[j], x, residuals[j], len, 0.2 * len);
    forward_local_linear(residuals_yhats[j], local_residual_params[j], x, len);
  }

  write_csv("initial_yhats.csv", yhats, "yhat0,yhat1,yhat2", 3, len, false);
  write_csv("residuals.csv", residuals, "residuals0,residuals1,residual2", 3,
            len, false);
  write_csv("residuals_yhats.csv", residuals_yhats,
            "residuals0,residuals1,residual2", 3, len, false);

  // Allocate storage for span selection
  float_t *selected_spans = malloc(sizeof(float_t) * len);
  for (size_t j = 0; j < len; j++) {
    // Single point residual for the 3 spans 0.05, 0.2, 0.5
    float_t residuals_j[] = {fabs(residuals_yhats[0][j]),
                             fabs(residuals_yhats[1][j]),
                             fabs(residuals_yhats[2][j])};

    size_t smallest_k = 0;
    for (size_t k = 0; k < 3; k++) {
      printf("reiduals_j[%lu]: %lf\n", k, residuals_j[k]);
      printf("smallest_k[%lu]: %lf\n", k, residuals_j[smallest_k]);
      if (residuals_j[k] < residuals_j[smallest_k]) {
        smallest_k = k;
      }
    }
    printf("final smallest_k: %lu\n", smallest_k);


    // Store the selected span for this point
    selected_spans[j] = spans[smallest_k];

  }

  // Smooth the selected spans using midrange smoother
  Params *span_smooth_params = malloc(sizeof(Params) * len);
  float_t *smoothed_spans = malloc(sizeof(float_t) * len);
  fit_local_linear(span_smooth_params, x, selected_spans, len, 0.2 * len);
  forward_local_linear(smoothed_spans, span_smooth_params, x, len);

  // Final loop to do interpolation using smoothed spans
  float_t *interpolated_yhat = malloc(sizeof(float_t) * len);
  for (size_t j = 0; j < len; j++) {
      float_t f = smoothed_spans[j] - spans[1];  // Compare to midrange (0.2)
      
      printf("diff f: %lf\n", f);
      if (f < 0) {
          // Interpolate between tweeter and midrange
          f = -f/(spans[1] - spans[0]);
          printf("f: %lf\n", f);
          interpolated_yhat[j] = (1.0 - f) * yhats[1][j] + f * yhats[0][j];
      } else {
          // Interpolate between midrange and woofer
          printf("f: %lf\n", f);
          f = f/(spans[2] - spans[1]);
          interpolated_yhat[j] = (1.0 - f) * yhats[1][j] + f * yhats[2][j];
      }
  }
  
  // Final smoothing of interpolated values using tweeter span
  Params *final_smooth_params = malloc(sizeof(Params) * len);
  float_t *final_smoothed = malloc(sizeof(float_t) * len);

  fit_local_linear(final_smooth_params, x, interpolated_yhat, len, 0.05 * len);
  forward_local_linear(final_smoothed, final_smooth_params, x, len);

  return final_smoothed;  // Return the final smoothed version instead of interpolated_yhat

}
