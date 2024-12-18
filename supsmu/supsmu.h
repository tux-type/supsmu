#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#pragma once

typedef struct RunningStats {
  float_t x_mean;
  float_t y_mean;
  float_t variance;
  float_t covariance;
  float_t sum_weight;
} RunningStats;

typedef struct SmoothState {
  // 0
  float_t y_tweeter;
  // 1
  float_t residual_tweeter;
  // 2
  float_t y_midrange;
  // 3
  float_t residual_midrange;
  // 4
  float_t y_woofer;
  // 5
  float_t residual_woofer;
  // 6
  float_t residual;
} SmoothState;

void supsmu(size_t n, float_t *x, float_t *y, float_t *w, int iper,
            float_t span, float_t bass, float_t *smo);
void write_csv(char *file_name, float_t **data, char *col_names,
               size_t num_cols, size_t num_rows, bool row_major);
void read_csv(char *file_name, float_t *x, float_t *y);
