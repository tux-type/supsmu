#include "supsmu.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })

void smooth(size_t n, double *x, double *y, double *w, double span, int iper,
            double vsmlsq, double *smo, double *acvr);

void update_stats(RunningStats *stats, double x, double y, double weight,
                  bool adding);

void supsmu(size_t n, double *x, double *y, double *w, int iper, double span,
            double bass, double *smo);

// TODO: Refactor iper to be a boolean and add another flag for negative iper
void supsmu(size_t n, double *x, double *y, double *w, int iper, double span,
            double alpha, double *smo) {
  // Change sc to an array of structs containing 7 fields?
  // -- No, because sometimes we loop over n
  // -- Or perhaps do struct of 7 arrays? -- GOOD since Fortran arrays in
  // col-major order
  double *sc = calloc(n * 7, sizeof(double));
  double spans[3] = {0.05, 0.2, 0.5};
  double small = 1.0e-7;
  double eps = 1.0e-3;

  // Edge case: smoothed values as weighted mean of y
  if (x[n - 1] <= x[0]) {
    double sum_y = 0;
    double sum_w = 0;
    for (size_t j = 0; j < n; j++) {
      sum_y = sum_y + w[j] * y[j];
      sum_w = sum_w + w[j];
    }
    double a = sum_w > 0 ? a = sum_y / sum_w : 0;
    for (size_t j = 0; j < n; j++) {
      smo[j] = a;
    }
    return;
  }

  size_t i = n / 4; // Q1
  size_t j = 3 * i; // Q3
  // Offset by 1 to account for 0 based indexing
  double scale = x[j - 1] - x[i - 1]; // Scale = IQR

  // TODO: Double check if this can enter an infinite loop e.g. when x values
  // are same (shouldn't happen due to bounds check above)
  while (scale <= 0) {
    j = j < (n - 1) ? j + 1 : j;
    i = i > 0 ? i - 1 : i;
    scale = x[j] - x[i];
  }

  double vsmlsq = pow(eps * scale, 2);
  size_t jper = iper;

  jper = (iper == 2 && (x[0] < 0.0 || x[n] > 1.0)) ? 1 : jper;
  jper = (jper < 1 || jper > 2) ? 1 : jper;

  //
  // Using provided span
  //
  if (span > 0.0) {
    // Call with full scratch memory size
    smooth(n, x, y, w, span, jper, vsmlsq, smo, sc);
    return;
  }

  //
  // Using cross validation for three spans
  //
  size_t row = 0;
  size_t col = 0;
  // DO NOT USE (Row major) Index:  row_i * num_columns + col_j
  // (Col major) Index: col_j * num_rows + row_i
  for (size_t i = 0; i < 3; i++) {
    col = 2 * i;
    smooth(n, x, y, w, spans[i], jper, vsmlsq, &sc[col * n + row],
           &sc[6 * n + row]);
    col = 2 * i + 1;
    // NULL since h should not be used in this second pass
    smooth(n, x, &sc[6 * n + row], w, spans[1], -jper, vsmlsq,
           &sc[col * n + row], NULL);
  }

  // Find optimal spans
  for (size_t j = 0; j < n; j++) {
    double resmin = DBL_MAX;

    // Find spans with lowest residuals
    row = j;
    for (size_t i = 0; i < 3; i++) {
      col = 2 * i + 1;
      if (sc[col * n + row] < resmin) {
        resmin = sc[col * n + row];
        sc[6 * n + row] = spans[i];
      }
    }

    // Alpha/bass adjustment
    if (alpha > 0.0 && alpha <= 10.0 && resmin < sc[5 * n + row] &&
        resmin > 0.0) {
      // TODO: CLEAN UP THIS MESS
      sc[6 * n + row] =
          sc[6 * n + row] +
          (spans[2] - sc[6 * n + row]) *
              pow(max(small, resmin / sc[5 * n + row]), 10.0 - alpha);
    }
  }

  // Smooth spans
  smooth(n, x, &sc[6 * n + 0], w, spans[1], -jper, vsmlsq, &sc[1 * n + 0],
         NULL);

  // Interpolate between residuals (?)
  for (size_t j = 0; j < n; j++) {
    row = j;
    if (sc[1 * n + row] <= spans[0]) {
      sc[1 * n + row] = spans[0];
    }
    if (sc[1 * n + row] >= spans[2]) {
      sc[1 * n + row] = spans[2];
    }
    double f = sc[1 * n + row] - spans[1];
    if (f >= 0.0) {
      f = f / (spans[2] - spans[1]);
      // Index 2 - midrange, Index 0 - tweeter
      sc[3 * n + row] = (1.0 - f) * sc[2 * n + row] + f * sc[4 * n + row];
    } else {
      f = -f / (spans[1] - spans[0]);
      // Index 2 - midrange, Index 4 - woofer
      sc[3 * n + row] = (1.0 - f) * sc[2 * n + row] + f * sc[0 * n + row];
    }
  }

  smooth(n, x, &sc[3 * n + 0], w, spans[0], -jper, vsmlsq, smo, NULL);

  free(sc);
}

void smooth(size_t n, double *x, double *y, double *w, double span, int iper,
            double vsmlsq, double *smo, double *acvr) {
  // w: weights or arange(1, n)
  // TODO: iper SHOULD BE BOOL
  // iper: periodic variable flag
  //   iper=1 => x is ordered interval variable
  //   iper=2 => x is a periodic variable with values
  //             in the range (0.0, 1.0) and period 1.0
  //  span: span of 0 indicates cross-validation/automatic selection
  //  alpha: controls small span (high freq.) penalty used with automatic
  //         span selection (0.0 < alpha <= 10.0)
  // vsmlsq: ??
  // smo: smooth output?

  int jper = abs(iper);
  // J: window size/span
  size_t half_J = floor(0.5 * span * n + 0.5);
  half_J = half_J < 2 ? 2 : half_J;
  size_t J = 2 * half_J + 1;
  J = J > n ? n : J;

  RunningStats stats = {0};

  // Initial fill of the window
  for (size_t i = 0; i < J; i++) {
    ssize_t j = jper == 2 ? i - half_J - 1 : i;
    // TODO: Tidy up, split out if statement for only the jper == 2 case
    // j is purely for periodic case, when jper is not 2, j should always == i
    // and >= 0
    double x_j;
    if (j >= 0) {
      x_j = x[j];
    } else {
      j = n + j;
      // Adjust by -1 so x appears close to other points in the window (?)
      x_j = x[j] - 1.0;
    }
    // TODO: Determine if it's worthwile to in-line update_stats
    update_stats(&stats, x_j, y[j], w[j], true);
  }

  // Main smoothing loop
  for (size_t j = 0; j < n; j++) {
    // window: (out < i < in) <- might not always be true on window edges

    // Remove point falling out of window (if exists)
    // TODO: Rename out to start; in to end
    ssize_t out = j - half_J - 1;
    size_t in = j + half_J;

    // TODO: Further tidy up of conditional statements
    if (jper == 2 || (out >= 0 && in < n)) {
      double x_out;
      double x_in;
      // Out of bounds checks: only applies when jper == 2
      if (out < 0) {
        out = n + out;
        x_out = x[out] - 1.0;
        x_in = x[in];
      } else if (in >= n) {
        in = in - n;
        x_out = x[out];
        x_in = x[in] + 1.0;
      } else {
        x_out = x[out];
        x_in = x[in];
      }
      update_stats(&stats, x_out, y[out], w[out], false);
      update_stats(&stats, x_in, y[in], w[in], true);
    }

    // TODO: Rename to something more reasonable
    double a = 0.0;
    // TODO: Figure out what is vsmlsq?
    if (stats.variance > vsmlsq) {
      a = stats.covariance / stats.variance;
    }
    smo[j] = a * (x[j] - stats.x_mean) + stats.y_mean;

    // TODO: Refactor this case where iper <= 0, as this is using a "hack"
    // This is calculating the cross validation residual for each point
    // Smaller CV values => better fit
    if (iper > 0 && acvr != NULL) {
      double h = 0;
      h = stats.sum_weight > 0 ? 1.0 / stats.sum_weight : h;
      h = stats.variance > vsmlsq
              ? h + pow((x[j] - stats.x_mean), 2) / stats.variance
              : h;
      acvr[j] = 0.0;
      a = 1.0 - w[j] * h;
      if (a > 0) {
        acvr[j] = fabs(y[j] - smo[j]) / a;
        // TODO: WHAT IS GOING ON HERE???
      } else if (j > 0) {
        acvr[j] = acvr[j - 1];
      }
    }
  }

  // Recompute fitted values smo[j] as weighted mean for non-unique x[.] values:
  for (size_t j = 0; j < n; j++) {
    size_t j0 = j;
    double sum_y = smo[j] * w[j];
    double sum_weight = w[j];

    while (j < (n - 1) && x[j + 1] <= x[j]) {
      j = j + 1;
      sum_y = sum_y + w[j] * smo[j];
      sum_weight = sum_weight + w[j];
    }
    if (j > j0) {
      double a = sum_weight > 0 ? sum_y / sum_weight : 0;
      for (size_t i = j0; i <= j; i++) {
        smo[i] = a;
      }
    }
  }
}

void update_stats(RunningStats *stats, double x, double y, double weight,
                  bool adding) {
  // Adding: adding or removing points to/from window

  double sum_weight_original = stats->sum_weight;

  if (adding) {
    stats->sum_weight += weight;
  } else {
    stats->sum_weight -= weight;
  }

  if (stats->sum_weight > 0) {
    // Update means
    if (adding) {
      stats->x_mean = (sum_weight_original * stats->x_mean + weight * x) /
                      stats->sum_weight;
      stats->y_mean = (sum_weight_original * stats->y_mean + weight * y) /
                      stats->sum_weight;
    } else {
      stats->x_mean = (sum_weight_original * stats->x_mean - weight * x) /
                      stats->sum_weight;
      stats->y_mean = (sum_weight_original * stats->y_mean - weight * y) /
                      stats->sum_weight;
    }
  }

  double tmp = 0;
  // sum_weight_original = 0 for the initial point, since variance is
  // 0 for a single vale
  if (sum_weight_original > 0) {
    tmp =
        stats->sum_weight * weight * (x - stats->x_mean) / sum_weight_original;
  }
  if (adding) {
    stats->variance += tmp * (x - stats->x_mean);
    stats->covariance += tmp * (y - stats->y_mean);
  } else {
    stats->variance -= tmp * (x - stats->x_mean);
    stats->covariance -= tmp * (y - stats->y_mean);
  }
}
