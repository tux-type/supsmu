#include "local.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 6

void check_alloc(void *p, size_t len) {
  if (p == NULL) {
    printf("Failed to allocate %zu length array\n", len);
    exit(EXIT_FAILURE);
  }
}

void test_fit_local_linear(void) {

  // POSITIVE SLOPE
  float_t x1[] = {0, 1, 2, 3, 4, 5};
  float_t y1[] = {5, 4, 3, 2, 1, 0};
  int j1 = 2;
  Params *local_params1 = malloc(sizeof(Params) * N);
  check_alloc(local_params1, N);

  fit_local_linear(local_params1, x1, y1, 6, j1);

  for (size_t i = 0; i < 6; i++) {
    assert(fabs(local_params1[i].w - (- 1)) < 0.01);
    assert(fabs(local_params1[i].b - 5) < 0.01);
  }
  printf("Test passed: Local Linear Positive Slope\n");
  free(local_params1);

  // NEGATIVE SLOPE
  float_t x2[6] = {0, 1, 2, 3, 4, 5};
  float_t y2[6] = {0, 1, 2, 3, 4, 5};
  int j2 = 2;
  Params *local_params2 = malloc(sizeof(Params) * N);
  check_alloc(local_params2, N);

  fit_local_linear(local_params2, x2, y2, 6, j2);

  for (size_t i = 0; i < 6; i++) {
    assert(fabs(local_params2[i].w - 1) < 0.01);
    assert(fabs(local_params2[i].b - 0) < 0.01);
  }
  printf("Test passed: Local Linear Negative Slope\n");
  free(local_params2);

  // ZERO SLOPE
  float_t x3[6] = {0, 1, 2, 3, 4, 5};
  float_t y3[6] = {1, 1, 1, 1, 1, 1};
  int j3 = 2;
  Params *local_params3 = malloc(sizeof(Params) * N);
  check_alloc(local_params3, N);

  fit_local_linear(local_params3, x3, y3, 6, j3);

  for (size_t i = 0; i < 6; i++) {
    assert(fabs(local_params3[i].w - 0) < 0.01);
    assert(fabs(local_params3[i].b - 1) < 0.01);
  }
  printf("Test passed: Local Linear Zero Slope\n");
  free(local_params3);

  // ODD LENGTH J
  float_t x4[6] = {0, 1, 2, 3, 4, 5};
  float_t y4[6] = {1, 1, 1, 1, 1, 1};
  int j4 = 3;
  Params *local_params4 = malloc(sizeof(Params) * N);
  check_alloc(local_params4, N);

  fit_local_linear(local_params4, x4, y4, 6, j4);

  for (size_t i = 0; i < 6; i++) {
    assert(fabs(local_params4[i].w - 0) < 0.01);
    assert(fabs(local_params4[i].b - 1) < 0.01);
  }
  printf("Test passed: Local Linear Odd Length J\n");
  free(local_params4);


}

int main(void) {
  test_fit_local_linear();
  return 0;
}
