#include "supsmu.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef enum Span {
  TWEETER,
  MIDRANGE,
  WOOFER,
} Span;

typedef enum CVFields {
  Y_TWEETER,         // 0
  RESIDUAL_TWEETER,  // 1
  Y_MIDRANGE,        // 2
  RESIDUAL_MIDRANGE, // 3
  Y_WOOFER,          // 4
  RESIDUAL_WOOFER,   // 5
  TEMP_RESIDUAL,     // 6
} CVFields;

typedef enum SpanFields {
  BEST_SPANS = 6,
  BEST_SPANS_MIDRANGE = 1,
  Y_INTERPOLATED = 3,
} SpanFields;

typedef struct SmoothState {
  double *data;
  size_t n;
} SmoothState;

typedef struct RunningStats {
  double x_mean;
  double y_mean;
  double variance;
  double covariance;
  double sum_weight;
} RunningStats;

inline double *get_field(const SmoothState *ss, const size_t field_idx,
                         const size_t sample_idx) {
  return &ss->data[field_idx * ss->n + sample_idx];
}

inline double max(double a, double b) { return a > b ? a : b; }

void smooth(size_t n, const double *x, const double *y, const double *w, double span, int iper,
            double vsmlsq, double *smo, double *acvr);

void update_stats(RunningStats *stats, double x, double y, double weight,
                  bool adding);

// TODO: Refactor iper to be a boolean and add another flag for negative iper
void supsmu(size_t n, const double *x, const double *y, const double *w, int iper, double span,
            double alpha, double *smo, double *sc) {
  SmoothState ss = {.data = sc, .n = n};
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

  // Assert that the above early return worked as expected (state is valid).
  assert(x[n - 1] > x[0]);
  while (scale <= 0) {
    j = j < (n - 1) ? j + 1 : j;
    i = i > 0 ? i - 1 : i;
    scale = x[j] - x[i];
  }

  double vsmlsq = pow(eps * scale, 2);
  size_t jper = iper;

  jper = (iper == 2 && (x[0] < 0.0 || x[n] > 1.0)) ? 1 : jper;
  jper = (jper < 1 || jper > 2) ? 1 : jper;

  // Using provided span
  if (span > 0.0) {
    // Call with full scratch memory size
    smooth(n, x, y, w, span, jper, vsmlsq, smo, sc);
    return;
  }

  // Using cross validation for three spans
  CVFields y_idx[3] = {Y_TWEETER, Y_MIDRANGE, Y_WOOFER};
  CVFields residual_idx[3] = {RESIDUAL_TWEETER, RESIDUAL_MIDRANGE,
                              RESIDUAL_WOOFER};

  for (Span sp = TWEETER; sp <= WOOFER; sp++) {
    smooth(n, x, y, w, spans[sp], jper, vsmlsq, get_field(&ss, y_idx[sp], 0),
           get_field(&ss, TEMP_RESIDUAL, 0));
    // NULL since h should not be used in this second pass
    smooth(n, x, get_field(&ss, TEMP_RESIDUAL, 0), w, spans[MIDRANGE], -jper,
           vsmlsq, get_field(&ss, residual_idx[sp], 0), NULL);
  }

  // Find optimal spans
  for (size_t row = 0; row < n; row++) {
    double resmin = DBL_MAX;

    // Find spans with lowest residuals
    for (Span sp = TWEETER; sp <= WOOFER; sp++) {
      if (*get_field(&ss, residual_idx[sp], row) < resmin) {
        resmin = *get_field(&ss, residual_idx[sp], row);
        *get_field(&ss, BEST_SPANS, row) = spans[sp];
      }
    }

    // Alpha/bass adjustment
    if (alpha > 0.0 && alpha <= 10.0 &&
        resmin < *get_field(&ss, RESIDUAL_WOOFER, row) && resmin > 0.0) {
      double *best_spans = get_field(&ss, BEST_SPANS, row);
      double *residual_woofer = get_field(&ss, RESIDUAL_WOOFER, row);
      *best_spans = *best_spans + (spans[WOOFER] - *best_spans) *
                                      pow(max(small, resmin / *residual_woofer),
                                          10.0 - alpha);
    }
  }

  // Smooth spans
  smooth(n, x, get_field(&ss, BEST_SPANS, 0), w, spans[MIDRANGE], -jper, vsmlsq,
         get_field(&ss, BEST_SPANS_MIDRANGE, 0), NULL);

  // Interpolate between y values based on residuals
  for (size_t row = 0; row < n; row++) {
    if (*get_field(&ss, BEST_SPANS_MIDRANGE, row) <= spans[TWEETER]) {
      *get_field(&ss, BEST_SPANS_MIDRANGE, row) = spans[TWEETER];
    }
    if (*get_field(&ss, BEST_SPANS_MIDRANGE, row) >= spans[WOOFER]) {
      *get_field(&ss, BEST_SPANS_MIDRANGE, row) = spans[WOOFER];
    }
    double f = *get_field(&ss, BEST_SPANS_MIDRANGE, row) - spans[MIDRANGE];
    if (f >= 0.0) {
      f = f / (spans[WOOFER] - spans[MIDRANGE]);
      // Index 2 - midrange, Index 4 - woofer
      *get_field(&ss, Y_INTERPOLATED, row) =
          (1.0 - f) * *get_field(&ss, Y_MIDRANGE, row) +
          f * *get_field(&ss, Y_WOOFER, row);
    } else {
      f = -f / (spans[MIDRANGE] - spans[TWEETER]);
      // Index 2 - midrange, Index 0 - tweeter
      *get_field(&ss, Y_INTERPOLATED, row) =
          (1.0 - f) * *get_field(&ss, Y_MIDRANGE, row) +
          f * *get_field(&ss, Y_TWEETER, row);
    }
  }

  smooth(n, x, get_field(&ss, Y_INTERPOLATED, 0), w, spans[TWEETER], -jper,
         vsmlsq, smo, NULL);
}

void smooth(size_t n, const double *x, const double *y, const double *w, double span, int iper,
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
  // Separate loops to allow optimisations on non-periodic case
  if (jper == 1) {
    for (size_t i = 0; i < J; i++) {
      update_stats(&stats, x[i], y[i], w[i], true);
    }
  } else if (jper == 2) {
    for (size_t i = 0; i < J; i++) {
      ssize_t j = i - half_J - 1;
      double x_j;

      if (j < 0) {
        // Wrap around and adjust by -1 so x appears close to other points in
        // the window
        j = n + j;
        x_j = x[j] - 1.0;
      } else {
        x_j = x[j];
      }
      update_stats(&stats, x_j, y[j], w[j], true);
    }
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
      } else if (j > 0) {
        acvr[j] = acvr[j - 1];
      }
    }
  }

  // Recompute fitted values smo[j] as weighted mean for non-unique x values:
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
  // 0 for a single value
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
