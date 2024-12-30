#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#pragma once

typedef struct RunningStats {
  double x_mean;
  double y_mean;
  double variance;
  double covariance;
  double sum_weight;
} RunningStats;

typedef struct SmoothState {
  // 0
  double y_tweeter;
  // 1
  double residual_tweeter;
  // 2
  double y_midrange;
  // 3
  double residual_midrange;
  // 4
  double y_woofer;
  // 5
  double residual_woofer;
  // 6
  double residual;
} SmoothState;

void supsmu(size_t n, double *x, double *y, double *w, int iper,
            double span, double bass, double *smo);
void write_csv(char *file_name, double **data, char *col_names,
               size_t num_cols, size_t num_rows, bool row_major);
void read_csv(char *file_name, double *x, double *y);
