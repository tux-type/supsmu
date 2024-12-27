import warnings

import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "supsmu.h":
    void c_supsmu "supsmu"(size_t n, float *x, float *y, float *w, int iper,
                float span, float bass, float *smo)


def supsmu(np.ndarray[DTYPE_t, ndim=1] x,
           np.ndarray[DTYPE_t, ndim=1] y,
           object wt=None,
           float span=0,
           bint periodic=False,
           float bass=0):
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


    if periodic:
        iper = 2
        if x.min() < 0 or x.max() > 1:
            raise ValueError("x must be between 0 and 1 when periodic")
    else:
        iper = 1
    
    size = y.shape[0]
    

    # Ensure arrays are contiguous
    x = np.ascontiguousarray(x, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DTYPE)

    # C type for wt_arr array
    cdef np.ndarray[DTYPE_t, ndim=1] wt_arr

    if wt is None:
        wt_arr = np.ones(size, dtype=DTYPE)
    elif not np.issubdtype(wt.dtype, np.number):
        raise ValueError("wt_arr should be a numeric array")
    else:
        wt_arr = np.ascontiguousarray(wt, dtype=DTYPE)

    if x.shape[0] != size:
        raise ValueError("x and y arrays should be the same length")
    if wt_arr.shape[0] != size:
        raise ValueError("weight and y arrays should be the same length")


    # Create output array
    cdef np.ndarray[DTYPE_t, ndim=1] smo = np.empty_like(x, dtype=DTYPE)
  

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(wt_arr)
    if not np.any(finite):
        raise ValueError("x, y, and wt_arr must have some finite observations (not nan or inf)")
    elif not np.all(finite):
        warnings.warn(f"Warning: dropped {size - sum(finite)} non-finite observaitons")
        finite_idx = np.where(finite)
        x = x[finite_idx]
        y = y[finite_idx]
        wt_arr = wt_arr[finite_idx]

    cdef size_t n = y.shape[0]

    # TODO: Cast to np.float64

    c_supsmu(
        n,
        <float*>x.data,
        <float*>y.data,
        <float*>wt_arr.data,
        iper,
        span,
        bass,
        <float*>smo.data
    )
    
    return smo
