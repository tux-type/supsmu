suppressMessages(library("dplyr"))

supsmu_expected <- function(file_name) {
  data = read.csv(file_name, stringsAsFactors=FALSE)
  
  expected = supsmu(data$x, data$y_noisy)
  stopifnot(all(expected$x == data$x))
  data$y_expected = expected$y
  
  write.csv(data, file=file_name, row.names=FALSE)
}

supsmu_expected("./tests/data/test_sin.csv")
supsmu_expected("./tests/data/test_mixed_freq_sin.csv")
supsmu_expected("./tests/data/test_mixed_noise_sin.csv")
supsmu_expected("./tests/data/test_complex_sin.csv")
