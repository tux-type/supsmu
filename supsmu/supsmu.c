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

const char *get_field(char *line, int num) {
  const char *tok;
  for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n")) {
    if (!--num) {
      return tok;
    }
  }
  return NULL;
}

void read_csv(char *file_name, float_t *x, float_t *y) {
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    printf("File %s could not be opened for reading.", file_name);
    exit(EXIT_FAILURE);
  }
  char line[1024];
  size_t i = 0;
  while (fgets(line, 1024, fp)) {
    char *tmp_x = strdup(line);
    char *tmp_y = strdup(line);
    if (i == 0) {
      i++;
      continue;
    }
    x[i - 1] = strtof(get_field(tmp_x, 1), NULL);
    y[i - 1] = strtof(get_field(tmp_y, 2), NULL);
    i++;
    free(tmp_x);
    free(tmp_y);
  }
  fclose(fp);
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

void smooth(size_t n, float_t *x, float_t *y, float_t *w, float_t span,
            int iper, float_t vsmlsq, float_t *smo, float_t *acvr);

void update_stats(RunningStats *stats, float_t x, float_t y, float_t weight,
                  bool adding);

void supsmu(size_t n, float_t *x, float_t *y, float_t *w, int iper,
            float_t span, float_t bass, float_t *smo);

// TODO: Refactor iper to be a boolean and add another flag for negative iper
void supsmu(size_t n, float_t *x, float_t *y, float_t *w, int iper,
            float_t span, float_t alpha, float_t *smo) {
  // Change sc to an array of structs containing 7 fields?
  // -- No, because sometimes we loop over n
  // -- Or perhaps do struct of 7 arrays? That way we can loopdy loop over them
  // init? -- GOOD since Fortran arrays in col-major order
  float_t *sc = calloc(n * 7, sizeof(float_t));
  float_t spans[3] = {0.05, 0.2, 0.5};
  float_t small = 1.0e-7;
  float_t eps = 1.0e-3;

  size_t q1 = n / 4;
  size_t q3 = 3 * q1;
  float_t iqr = x[q3] - x[q1];

  // TODO: Add handling for when iqr <= 0 (i.e. when x[qr] < x[q1])

  float_t vsmlsq = pow(eps * iqr, 2);
  size_t jper = iper;

  jper = (iper == 2 && (x[0] < 0.0 || x[n] > 1.0)) ? 1 : jper;
  jper = (jper < 1 || jper > 2) ? 1 : jper;

  //
  // Using provided span
  //
  if (span > 0.0) {
    // Call with full scratch memory size
    printf("Smoothing with a set span, not CV!");
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
    // printf("y: %lf, %lf, %lf, %lf, %lf\n", y[0], y[1], y[2], y[3], y[4]);
    // printf("sc[6 * n + row]: %lf, %lf, %lf, %lf, %lf\n", (&sc[6 * n +
    // row])[0],
    //        (&sc[6 * n + row])[1], (&sc[6 * n + row])[2], (&sc[6 * n +
    //        row])[3],
    //        (&sc[6 * n + row])[4]);
    col = 2 * i + 1;
    // NULL since h should not be used in this second pass
    smooth(n, x, &sc[6 * n + row], w, spans[1], -jper, vsmlsq,
           &sc[col * n + row], NULL);
  }

  // Find optimal spans
  for (size_t j = 0; j < n; j++) {
    // Perhaps change all float_t types to double, as technically float_t could
    // be double in which case FLT_MAX would be smaller than max value of
    // float_t.
    float_t resmin = FLT_MAX;

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

  // printf("sc[6 * n + 0]: %lf, %lf, %lf, %lf, %lf\n", (&sc[6 * n + 0])[0],
  //        (&sc[6 * n + 0])[1], (&sc[6 * n + 0])[2], (&sc[6 * n + 0])[3],
  //        (&sc[6 * n + 0])[4]);
  // printf("sc[6 * n + 0]: %lf, %lf, %lf, %lf, %lf\n", (&sc[1 * n + 0])[0],
  //        (&sc[1 * n + 0])[1], (&sc[1 * n + 0])[2], (&sc[1 * n + 0])[3],
  //        (&sc[1 * n + 0])[4]);
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
    float_t f = sc[1 * n + row] - spans[1];
    printf("f: %lf", f);
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
}

void smooth(size_t n, float_t *x, float_t *y, float_t *w, float_t span,
            int iper, float_t vsmlsq, float_t *smo, float_t *acvr) {
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

  printf("y: %lf, %lf, %lf, %lf, %lf\n", y[0], y[1], y[2], y[3], y[4]);
  // Initial fill of the window
  for (size_t i = 0; i < J; i++) {
    size_t j = jper == 2 ? i - half_J - 1 : i;
    // TODO: Tidy up, split out if statement for only the jper == 2 case
    // j is purely for periodic case, when jper is not 2, j should always == i and >= 0
    float_t x_j;
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
  printf("xm: %lf\n", stats.x_mean);
  printf("ym: %lf\n", stats.y_mean);
  printf("var: %lf\n", stats.variance);
  printf("cvar: %lf\n", stats.covariance);

  // Main smoothing loop
  for (size_t j = 0; j < n; j++) {
    // window: (out < i < in) <- might not always be true on window edges

    // Remove point falling out of window (if exists)
    // TODO: Rename out to start; in to end
    ssize_t out = j - half_J - 1;
    size_t in = j + half_J;

    // TODO: Further tidy up of conditional statements
    if (jper == 2 || (out >= 0 && in < n)) {
      float_t x_out;
      float_t x_in;
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
      // printf("out: %lu\n", out);
      // printf("in: %lu\n", in);
      // printf("x[out]: %lf\n", x[out]);
      // printf("x[in]: %lf\n", x[in]);
      // printf("xm: %lf\n", stats.x_mean);
      // printf("ym: %lf\n", stats.y_mean);
      // printf("var: %lf\n", stats.variance);
      // printf("cvar: %lf\n", stats.covariance);
      // exit(EXIT_FAILURE);
    }

    // TODO: Rename to something more reasonable
    float_t a = 0.0;
    // TODO: Figure out what is vsmlsq?
    if (stats.variance > vsmlsq) {
      a = stats.covariance / stats.variance;
    }
    smo[j] = a * (x[j] - stats.x_mean) + stats.y_mean;

    // TODO: Refactor this case where iper <= 0, as this is using a "hack"
    // This is calculating the cross validation residual for each point
    // Smaller CV values => better fit
    if (iper > 0 && acvr != NULL) {
      float_t h = 0;
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

  printf("smo: %lf, %lf, %lf, %lf, %lf\n", smo[0], smo[1], smo[2], smo[3],
         smo[4]);
  // Recompute fitted values smo[j] as weighted mean for non-unique x[.] values:
  // TODO: Add code to handle same x values
}

void update_stats(RunningStats *stats, float_t x, float_t y, float_t weight,
                  bool adding) {
  // Adding: adding or removing points to/from window

  float_t sum_weight_original = stats->sum_weight;

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

  float_t tmp = 0;
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
