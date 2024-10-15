#include "local.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_fit_local_linear(void) {

  // POSITIVE SLOPE
  float_t x1[6] = {0, 1, 2, 3, 4, 5};
  float_t y1[6] = {5, 4, 3, 2, 1, 0};
  int j1 = 2;
  Params *params1 = fit_local_linear(x1, y1, 6, j1);
  printf("w1: %f\n", params1->w);
  printf("b1: %f\n", params1->b);

  for (size_t i = 0; i < 6; i++) {
    assert(-0.01 < (params1->w - (-1)) && (params1->w - (-1)) < 0.01);
    assert(-0.01 < (params1->b - 5) && (params1->b - 5) < 0.01);
  }

  // NEGATIVE SLOPE
  float_t x2[6] = {0, 1, 2, 3, 4, 5};
  float_t y2[6] = {0, 1, 2, 3, 4, 5};
  int j2 = 2;
  Params *params2 = fit_local_linear(x2, y2, 6, j2);
  printf("w2: %f\n", params2->w);
  printf("b2: %f\n", params2->b);

  for (size_t i = 0; i < 6; i++) {
    assert(-0.01 < (params2->w - 1) && (params2->w - 1) < 0.01);
    assert(-0.01 < (params2->b - 0) && (params2->b - 0) < 0.01);
  }

  // ZERO SLOPE
  float_t x3[6] = {0, 1, 2, 3, 4, 5};
  float_t y3[6] = {1, 1, 1, 1, 1, 1};
  int j3 = 2;
  Params *params3 = fit_local_linear(x3, y3, 6, j3);
  printf("w2: %f\n", params3->w);
  printf("b2: %f\n", params3->b);

  for (size_t i = 0; i < 6; i++) {
    assert(-0.01 < (params3->w - 0) && (params3->w - 0) < 0.01);
    assert(-0.01 < (params3->b - 1) && (params3->b - 1) < 0.01);
  }
}

int main(void) {
  test_fit_local_linear();
  return 0;
}
