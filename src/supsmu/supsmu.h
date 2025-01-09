#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#pragma once


void supsmu(size_t n, const double *x, const double *y, const double *w, bool periodic,
            double span, double bass, double *smo, double *sc);
