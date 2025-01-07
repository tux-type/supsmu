import warnings

import numpy as np
cimport numpy as np
from cython cimport floating

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "supsmu.h":
    void c_supsmu "supsmu"(size_t n, const double *x, const double *y, const double *w, int iper,
                double span, double bass, double *smo, double *sc)


def supsmu(np.ndarray[floating, ndim=1] x,
           np.ndarray[floating, ndim=1] y,
           object wt=None,
           double span=0,
           bint periodic=False,
           double bass=0):
    """
    Python wrapper for supsmu function.
    
    Parameters:
        x: np.ndarray[float32] - x values
        y: np.ndarray[float32] - y values
        w: np.ndarray[float32] - weights
        span: float - smoothing span
        periodic: bool - periodicity flag
        bass: float - bass enhancement
    
    Returns:
        np.ndarray[float32] - smoothed values
    """
    if span < 0 or span > 1:
        raise ValueError("Span should be between 0 and 1.")

    if not all([np.issubdtype(x.dtype, np.number), np.issubdtype(y.dtype, np.number)]):
        raise ValueError("x and y should be numeric arrays")


    cdef np.ndarray[DTYPE_t, ndim=1] x_arr
    cdef np.ndarray[DTYPE_t, ndim=1] y_arr

    # Ensure arrays are contiguous
    x_arr = np.ascontiguousarray(x, dtype=DTYPE)
    y_arr = np.ascontiguousarray(y, dtype=DTYPE)


    if periodic:
        iper = 2
        if x_arr.min() < 0 or x_arr.max() > 1:
            raise ValueError("x must be between 0 and 1 when periodic")
    else:
        iper = 1
    
    cdef size_t size = y_arr.shape[0]
    

    # C type for wt_arr array
    cdef np.ndarray[DTYPE_t, ndim=1] wt_arr

    if wt is None:
        wt_arr = np.ones(size, dtype=DTYPE)
    elif not np.issubdtype(wt.dtype, np.number):
        raise ValueError("wt_arr should be a numeric array")
    else:
        wt_arr = np.ascontiguousarray(wt, dtype=DTYPE)

    if x_arr.shape[0] != size:
        raise ValueError("x and y arrays should be the same length")
    if wt_arr.shape[0] != size:
        raise ValueError("weight and y arrays should be the same length")


    # Create output array
    cdef np.ndarray[DTYPE_t, ndim=1] smo = np.empty_like(x_arr, dtype=DTYPE)
    # Create scratch memory array
    cdef np.ndarray[DTYPE_t, ndim=1] sc = np.empty(shape=(size * 7,), dtype=DTYPE)
  

    finite = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(wt_arr)
    if not np.any(finite):
        raise ValueError("x, y, and wt_arr must have some finite observations (not nan or inf)")
    elif not np.all(finite):
        warnings.warn(f"Warning: dropped {size - sum(finite)} non-finite observaitons")
        finite_idx = np.where(finite)
        x_arr = x_arr[finite_idx]
        y_arr = y_arr[finite_idx]
        wt_arr = wt_arr[finite_idx]


    c_supsmu(
        size,
        <double*>x_arr.data,
        <double*>y_arr.data,
        <double*>wt_arr.data,
        iper,
        span,
        bass,
        <double*>smo.data,
        <double*>sc.data
    )
    
    return smo
