#include "linear.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
  // POSITIVE SLOPE
  float_t x1[6] = {0, 1, 2, 3, 4, 5};
  float_t y1[6] = {5, 4, 3, 2, 1, 0};
  float_t w1 = 0;
  float_t b1 = 0;
  fit(x1, y1, &w1, &b1, 6);
  printf("w1: %f\n", w1);
  printf("b1: %f\n", b1);

  assert(-0.01 < (w1 - (-1)) && (w1 - (-1)) < 0.01);
  assert(-0.01 < (b1 - 5) && (b1 - 5) < 0.01);

  // NEGATIVE SLOPE
  float_t x2[6] = {0, 1, 2, 3, 4, 5};
  float_t y2[6] = {0, 1, 2, 3, 4, 5};
  float_t w2 = 0;
  float_t b2 = 0;
  fit(x2, y2, &w2, &b2, 6);
  printf("w2: %f\n", w2);
  printf("b2: %f\n", b2);

  assert(-0.01 < (w2 - 1) && (w2 - 1) < 0.01);
  assert(-0.01 < (b2 - 0) && (b2 - 0) < 0.01);

  // ZERO SLOPE
  float_t x3[6] = {0, 1, 2, 3, 4, 5};
  float_t y3[6] = {1, 1, 1, 1, 1, 1};
  float_t w3 = 0;
  float_t b3 = 0;
  fit(x3, y3, &w3, &b3, 6);
  printf("w2: %f\n", w3);
  printf("b2: %f\n", b3);

  assert(-0.01 < (w3 - 0) && (w3 - 0) < 0.01);
  assert(-0.01 < (b3 - 1) && (b3 - 1) < 0.01);

  return 0;
}
