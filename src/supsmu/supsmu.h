#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#pragma once

/**
 * Performs Friedman's SuperSmoother algorithm to smooth the data.
 * Automatically chooses the best smoothing span at each point using
 * cross-validation.
 *
 * @param n        Number of data points
 * @param x        Array of x values [size: n]
 * @param y        Array of y values [size: n]
 * @param w        Array of weights for each point [size: n]
 * @param periodic True if data is periodic, false otherwise
 * @param span     Smoothing span (0 for cross-validation, otherwise between 0
 * and 1)
 * @param alpha    Bass enhancement parameter (between 0 and 10) for increased
 * smoothness
 * @param smo      Output array for smoothed y values [size: n]
 * @param sc       Working memory array [size: n]
 *
 * @note          Arrays x, y, w must be pre-allocated with size n
 * @note          Output array smo must be pre-allocated with size n
 * @note          Working array sc must be pre-allocated with size n * 7
 */
void supsmu(size_t n, const double *x, const double *y, const double *w,
            bool periodic, double span, double bass, double *smo, double *sc);
