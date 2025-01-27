# Supsmu

Supsmu is an implementation of Friedman's SuperSmoother algorithm - a time series smoother that uses
cross-validation to automatically select optimal spans for local linear regression.

The package is written in C for computational efficiency, with Python bindings for use with NumPy arrays.

![A comparison of noisy data and its smoothed version using Supsmu](assets/smoothing_comparison.png "Smoothing Comparison")

## Installation

Install supsmu with:
```sh
pip install supsmu
```

## Python Example
Minimal example using dummy data:

```Python
import numpy as np
from supsmu import supsmu

x = np.linspace(0, 1, 100)
# Dummy data - a basic sine wave
y = np.sin(2 * 2 * np.pi * x)
noise = np.random.normal(0, 0.2, 100)
y_noisy = y + noise

y_smooth = supsmu(x, y_noisy, periodic=True)
```

## Additional Information
The C algorithm is implemented with the intent to closely match the outputs of the Fortran version
(available in R), however there may still exist some inconsistencies.

## Parameters
| PARAMETER | TYPE | DESCRIPTION |
|-----------|------|-------------|
| `x` | `np.ndarray[np.floating \| np.integer]` | x values |
| `y` | `np.ndarray[np.floating \| np.integer]` | y values |
| `wt` | `np.ndarray[np.floating \| np.integer] \| None` | weights |
| `span` | `float` | smoothing span (0 for cross-validation, otherwise between 0 and 1) |
| `periodic` | `bool` | True if data is periodic, False otherwise |
| `bass` | `float` | bass enhancement (between 0 and 10) for increased smoothness |


## References
[1] J. H. Friedman, "A Variable Span Smoother", SLAC National Accelerator Laboratory (SLAC),
Menlo Park, CA (United States), SLAC-PUB-3477; STAN-LCS-005, Oct. 1984. doi: 10.2172/1447470.
