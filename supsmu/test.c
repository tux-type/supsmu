#include "linear.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
  // POSITIVE SLOPE
  float_t x1[6] = {0, 1, 2, 3, 4, 5};
  float_t y1[6] = {5, 4, 3, 2, 1, 0};
  Params params1 = fit(x1, y1, 6);
  printf("w1: %f\n", params1.w);
  printf("b1: %f\n", params1.b);

  assert(-0.01 < (params1.w - (-1)) && (params1.w - (-1)) < 0.01);
  assert(-0.01 < (params1.b - 5) && (params1.b - 5) < 0.01);

  // NEGATIVE SLOPE
  float_t x2[6] = {0, 1, 2, 3, 4, 5};
  float_t y2[6] = {0, 1, 2, 3, 4, 5};
  Params params2 = fit(x2, y2, 6);
  printf("w2: %f\n", params2.w);
  printf("b2: %f\n", params2.b);

  assert(-0.01 < (params2.w - 1) && (params2.w - 1) < 0.01);
  assert(-0.01 < (params2.b - 0) && (params2.b - 0) < 0.01);

  // ZERO SLOPE
  float_t x3[6] = {0, 1, 2, 3, 4, 5};
  float_t y3[6] = {1, 1, 1, 1, 1, 1};
  Params params3 = fit(x3, y3, 6);
  printf("w2: %f\n", params3.w);
  printf("b2: %f\n", params3.b);

  assert(-0.01 < (params3.w - 0) && (params3.w - 0) < 0.01);
  assert(-0.01 < (params3.b - 1) && (params3.b - 1) < 0.01);

  return 0;
}
