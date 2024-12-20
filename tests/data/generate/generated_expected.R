suppressMessages(library("dplyr"))

supsmu_expected <- function(file_name, check_x_unique=TRUE) {
  data = read.csv(file_name, stringsAsFactors=FALSE)
  expected = supsmu(data$x, data$y_noisy)
  
  if (check_x_unique) {
  stopifnot(all(expected$x == data$x))
  }
  
  data$y_expected = expected$y
  
  write.csv(data, file=file_name, row.names=FALSE)
}

supsmu_expected("tests/data/test_sin_periodic.csv")
supsmu_expected("tests/data/test_complex_sin_periodic.csv")

supsmu_expected("tests/data/test_sin_aperiodic.csv")
supsmu_expected("tests/data/test_complex_sin_aperiodic.csv")

supsmu_expected("tests/data/test_all_x_same.csv", check_x_unique=FALSE)
supsmu_expected("tests/data/test_some_x_same.csv", check_x_unique=FALSE)
