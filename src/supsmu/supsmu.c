#include "supsmu.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef enum Span {
  Tweeter,
  Midrange,
  Woofer,
} Span;

typedef enum CVFields {
  YTweeter,
  ResidualTweeter,
  YMidrange,
  ResidualMidrange,
  YWoofer,
  ResidualWoofer,
  TempResidual,
} CVFields;

typedef enum SpanFields {
  BestSpans = 6,
  BestSpansMidrange = 1,
  YInterpolated = 3,
} SpanFields;

typedef struct SmoothState {
  double *data;
  size_t n;
} SmoothState;

typedef struct RunningStats {
  double x_mean;
  double y_mean;
  double x_variance;
  double covariance;
  double sum_weight;
} RunningStats;

inline double *get_field(const SmoothState *ss, const size_t field_idx,
                         const size_t sample_idx) {
  return &ss->data[field_idx * ss->n + sample_idx];
}

inline double max(double a, double b) { return a > b ? a : b; }

void smooth(size_t n, const double *x, const double *y, const double *w,
            double span, bool periodic, bool save_residual, double var_tol,
            double *smo, double *adj_residuals);

void update_stats(RunningStats *stats, double x, double y, double weight,
                  bool adding);

void supsmu(size_t n, const double *x, const double *y, const double *w,
            bool periodic, double span, double alpha, double *smo, double *sc) {
  SmoothState *ss = &(struct SmoothState){.data = sc, .n = n};
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

  // Data scale-aware variance threshold
  double var_tol = pow(eps * scale, 2);

  if (x[0] < 0.0 || x[n] > 1.0) {
    periodic = false;
  }

  // Using provided span
  if (span > 0.0) {
    // Call with full scratch memory size
    smooth(n, x, y, w, span, periodic, true, var_tol, smo, sc);
    return;
  }

  // Using cross validation for three spans
  CVFields y_idx[3] = {YTweeter, YMidrange, YWoofer};
  CVFields residual_idx[3] = {ResidualTweeter, ResidualMidrange,
                              ResidualWoofer};

  for (Span sp = Tweeter; sp <= Woofer; sp++) {
    smooth(n, x, y, w, spans[sp], periodic, true, var_tol,
           get_field(ss, y_idx[sp], 0), get_field(ss, TempResidual, 0));
    // NULL since h should not be used in this second pass
    smooth(n, x, get_field(ss, TempResidual, 0), w, spans[Midrange], periodic,
           false, var_tol, get_field(ss, residual_idx[sp], 0), NULL);
  }

  // Find optimal spans
  for (size_t row = 0; row < n; row++) {
    double resmin = DBL_MAX;

    // Find spans with lowest residuals
    for (Span sp = Tweeter; sp <= Woofer; sp++) {
      if (*get_field(ss, residual_idx[sp], row) < resmin) {
        resmin = *get_field(ss, residual_idx[sp], row);
        *get_field(ss, BestSpans, row) = spans[sp];
      }
    }

    // Alpha/bass adjustment
    if (alpha > 0.0 && alpha <= 10.0 &&
        resmin < *get_field(ss, ResidualWoofer, row) && resmin > 0.0) {
      double *best_span = get_field(ss, BestSpans, row);
      double *residual_woofer = get_field(ss, ResidualWoofer, row);
      *best_span = *best_span +
                   (spans[Woofer] - *best_span) *
                       pow(max(small, resmin / *residual_woofer), 10.0 - alpha);
    }
  }

  // Smooth spans
  smooth(n, x, get_field(ss, BestSpans, 0), w, spans[Midrange], periodic, false,
         var_tol, get_field(ss, BestSpansMidrange, 0), NULL);

  // Interpolate between y values based on residuals
  for (size_t row = 0; row < n; row++) {
    if (*get_field(ss, BestSpansMidrange, row) <= spans[Tweeter]) {
      *get_field(ss, BestSpansMidrange, row) = spans[Tweeter];
    }
    if (*get_field(ss, BestSpansMidrange, row) >= spans[Woofer]) {
      *get_field(ss, BestSpansMidrange, row) = spans[Woofer];
    }
    double f = *get_field(ss, BestSpansMidrange, row) - spans[Midrange];
    if (f >= 0.0) {
      f = f / (spans[Woofer] - spans[Midrange]);
      // Index 2 - midrange, Index 4 - woofer
      *get_field(ss, YInterpolated, row) =
          (1.0 - f) * *get_field(ss, YMidrange, row) +
          f * *get_field(ss, YWoofer, row);
    } else {
      f = -f / (spans[Midrange] - spans[Tweeter]);
      // Index 2 - midrange, Index 0 - tweeter
      *get_field(ss, YInterpolated, row) =
          (1.0 - f) * *get_field(ss, YMidrange, row) +
          f * *get_field(ss, YTweeter, row);
    }
  }

  smooth(n, x, get_field(ss, YInterpolated, 0), w, spans[Tweeter], periodic,
         false, var_tol, smo, NULL);
}

/**
 * Performs smoothing using a set span and calculates adjusted CV residual if
 * needed
 *
 * @param n              Number of data points
 * @param x              Array of x values [size: n]
 * @param y              Array of y values [size: n]
 * @param w              Array of weights for each point [size: n]
 * @param span           Smoothing span (window width as fraction of total
 * points)
 * @param periodic       True if data is periodic, false otherwise
 * @param save_residual  True to compute adjusted residuals, false otherwise
 * @param var_tol        Variance tolerance for numerical stability
 * @param smo            Output array for smoothed y values [size: n]
 * @param adj_residuals  Output array for adjusted residuals if save_residual
 */
void smooth(size_t n, const double *x, const double *y, const double *w,
            double span, bool periodic, bool save_residual, double var_tol,
            double *smo, double *adj_residuals) {
  // J: window size/span
  size_t half_J = floor(0.5 * span * n + 0.5);
  half_J = half_J < 2 ? 2 : half_J;
  size_t J = 2 * half_J + 1;
  J = J > n ? n : J;

  RunningStats stats = {0};

  // Initial fill of the window
  // Separate loops to allow optimisations on non-periodic case
  if (periodic) {
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
  } else {
    for (size_t i = 0; i < J; i++) {
      update_stats(&stats, x[i], y[i], w[i], true);
    }
  }

  // Main smoothing loop
  for (size_t j = 0; j < n; j++) {
    // window: (out < i < in) <- except around window edges

    // Points falling out of window and being added to the window
    ssize_t out = j - half_J - 1;
    size_t in = j + half_J;

    if (out >= 0 && in < n) {
      update_stats(&stats, x[out], y[out], w[out], false);
      update_stats(&stats, x[in], y[in], w[in], true);
    } else if (periodic) {
      double x_out = x[out];
      double x_in = x[in];
      if (out < 0) {
        out = n + out;
        x_out = x[out] - 1.0;
      } else if (in >= n) {
        in = in - n;
        x_in = x[in] + 1.0;
      }
      update_stats(&stats, x_out, y[out], w[out], false);
      update_stats(&stats, x_in, y[in], w[in], true);
    }

    double slope = 0.0;
    if (stats.x_variance > var_tol) {
      slope = stats.covariance / stats.x_variance;
    }
    smo[j] = slope * (x[j] - stats.x_mean) + stats.y_mean;

    // Calculate the cross validation residual for each point (equivalent to
    // leave-one-out) https://robjhyndman.com/hyndsight/loocv-linear-models/
    // Smaller CV residual values => better fit
    // Rationale: the CV residuals indicate stability/smoothness: ignoring a
    //   single point should not throw off the rest of the predicted values
    if (save_residual && adj_residuals != NULL) {
      // aka h value of a Hat matrix
      double leverage = 0;
      leverage = stats.sum_weight > 0 ? 1.0 / stats.sum_weight : leverage;
      leverage =
          stats.x_variance > var_tol
              ? leverage + pow((x[j] - stats.x_mean), 2) / stats.x_variance
              : leverage;
      adj_residuals[j] = 0.0;
      double adj = 1.0 - w[j] * leverage;
      if (adj > 0) {
        adj_residuals[j] = fabs(y[j] - smo[j]) / adj;
      } else if (j > 0) {
        adj_residuals[j] = adj_residuals[j - 1];
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

/**
 * Updates running statistics for weighted data points
 *
 * Incrementally updates mean values, variance, and covariance statistics
 * for weighted (x,y) pairs. Can both add and remove points from the
 * running calculations, allowing for sliding window statistics.
 *
 * @param stats    Pointer to RunningStats structure holding statistical values
 * @param x        X value
 * @param y        Y value
 * @param weight   Weight associated with the data point
 * @param adding   True to add the point to statistics, false to remove it
 */
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

  double weighted_x_deviation = 0;
  // sum_weight_original = 0 for the initial point, since variance is
  // 0 for a single value
  if (sum_weight_original > 0) {
    weighted_x_deviation =
        stats->sum_weight * weight * (x - stats->x_mean) / sum_weight_original;
  }
  if (adding) {
    stats->x_variance += weighted_x_deviation * (x - stats->x_mean);
    stats->covariance += weighted_x_deviation * (y - stats->y_mean);
  } else {
    stats->x_variance -= weighted_x_deviation * (x - stats->x_mean);
    stats->covariance -= weighted_x_deviation * (y - stats->y_mean);
  }
}
