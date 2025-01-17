import warnings

import numpy as np
from numpy.typing import NDArray
cimport numpy as np
from cython cimport floating

np.import_array()

RealArray = NDArray[np.integer | np.floating]

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "supsmu.h":
    void c_supsmu "supsmu"(size_t n, const double *x, const double *y, const double *w,
                           bint periodic, double span, double bass, double *smo, double *sc)


def supsmu(
    x: RealArray,
    y: RealArray,
    wt: RealArray | None = None,
    span: int | float = 0,
    periodic: bool = False,
    bass: int | float = 0
) -> np.ndarray[DTYPE]:
    """
    Performs Friedman's SuperSmoother algorithm to smooth the data.
    Automatically chooses the best smoothing span at each point using
    cross-validation.
    
    Args:
        x: np.ndarray[np.floating | np.integer] - x values
        y: np.ndarray[np.floating | np.integer] - y values
        wt: np.ndarray[np.floating | np.integer] | None - weights
        span: float - smoothing span (0 for cross-validation, otherwise between 0 and 1)
        periodic: bool - True if data is periodic, False otherwise
        bass: float - bass enhancement (between 0 and 10) for increased smoothness


    Returns:
        np.ndarray[np.float64] - smoothed values
    """
    if span < 0 or span > 1:
        raise ValueError("Span should be between 0 and 1.")


    x_is_real = np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)
    y_is_real = np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.floating)

    if x_is_real and y_is_real:
        x = np.ascontiguousarray(x, dtype=DTYPE)
        y = np.ascontiguousarray(y, dtype=DTYPE)
    else:
        raise ValueError("x and y should be arrays containing real numbers")

    size = y.shape[0]

    if wt is None:
        wt = np.ascontiguousarray(np.ones(size, dtype=DTYPE))
    elif not (np.issubdtype(wt.dtype, np.integer) or np.issubdtype(wt.dtype, np.floating)):
        raise ValueError("wt should be an array containing real numbers")
    else:
        wt = np.ascontiguousarray(wt, dtype=DTYPE)


    if periodic and (x.min() < 0 or x.max() > 1):
        raise ValueError("x must be between 0 and 1 when periodic")

    if x.shape[0] != size:
        raise ValueError("x and y arrays should be the same length")
    if wt.shape[0] != size:
        raise ValueError("weight and y arrays should be the same length")

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(wt)
    if not np.any(finite):
        raise ValueError("x, y, and wt must have some finite observations (not nan or inf)")
    elif not np.all(finite):
        warnings.warn(f"Warning: dropped {size - sum(finite)} non-finite observaitons")
        finite_idx = np.where(finite)
        x = x[finite_idx]
        y = y[finite_idx]
        wt = wt[finite_idx]

    return _supsmu(x, y, wt, span, periodic, bass)

cdef np.ndarray[DTYPE_t, ndim=1] _supsmu(
    np.ndarray[DTYPE_t, ndim=1] x,
    np.ndarray[DTYPE_t, ndim=1] y,
    np.ndarray[DTYPE_t, ndim=1] wt,
    double span,
    bint periodic,
    double bass
):
    """
    Internal SuperSmooth function for interfacing with C.
    
    Args:
        x: np.ndarray[np.float64] - x values
        y: np.ndarray[np.float64] - y values
        wt: np.ndarray[np.float64] - weights
        span: float - smoothing span (0 for cross-validation, otherwise between 0 and 1)
        periodic: bool - True if data is periodic, False otherwise
        bass: float - bass enhancement (between 0 and 10) for increased smoothness


    Returns:
        np.ndarray[np.float64] - smoothed values
    """
    cdef np.ndarray[DTYPE_t, ndim=1] x_arr = np.ascontiguousarray(x)
    cdef np.ndarray[DTYPE_t, ndim=1] y_arr = np.ascontiguousarray(y)
    cdef np.ndarray[DTYPE_t, ndim=1] wt_arr = np.ascontiguousarray(wt)

    cdef size_t size = y_arr.shape[0]
    
    # Output and working arrays in Python to be GC'ed
    cdef np.ndarray[DTYPE_t, ndim=1] smo = np.empty_like(x_arr, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] sc = np.empty(shape=(size * 7,), dtype=DTYPE)

    c_supsmu(
        size,
        <double*>x_arr.data,
        <double*>y_arr.data,
        <double*>wt_arr.data,
        periodic,
        span,
        bass,
        <double*>smo.data,
        <double*>sc.data
    )
    
    return smo
