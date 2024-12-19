#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#pragma once

typedef struct RunningStats {
  float x_mean;
  float y_mean;
  float variance;
  float covariance;
  float sum_weight;
} RunningStats;

typedef struct SmoothState {
  // 0
  float y_tweeter;
  // 1
  float residual_tweeter;
  // 2
  float y_midrange;
  // 3
  float residual_midrange;
  // 4
  float y_woofer;
  // 5
  float residual_woofer;
  // 6
  float residual;
} SmoothState;

void supsmu(size_t n, float *x, float *y, float *w, int iper,
            float span, float bass, float *smo);
void write_csv(char *file_name, float **data, char *col_names,
               size_t num_cols, size_t num_rows, bool row_major);
void read_csv(char *file_name, float *x, float *y);
