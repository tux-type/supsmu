import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "supsmu.h":
    void c_supsmu "supsmu"(size_t n, float *x, float *y, float *w, int iper,
                float span, float bass, float *smo)


# def supersmoother(np.ndarray[DTYPE_t, ndim=1] arr):
#     n = np.shape[0]
#     y = np.zeros(n)
#     w = np.ones(n)
#     iper = 1
#     span = 0.0
#     bass = 0
#     smo = np.zeros(n)
#     if not arr.flags["C_CONTIGUOUS"]:
#         arr = np.ascontiguousarray(arr, dtype=DTYPE)
#
#     # TODO: Check if the nogil is actually needed
#     supsmu(n, <float*>arr.data, <float*>y.data, <float*>w.data, iper, span, bass, <float*>smo.data)
#
#     return smo


def supsmu(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
# def supsmu(np.ndarray[DTYPE_t, ndim=1] x,
#           np.ndarray[DTYPE_t, ndim=1] y,
#           np.ndarray[DTYPE_t, ndim=1] w,
#           int iper,
#           float span,
#           float bass):
    """
    Python wrapper for supsmu function.
    
    Parameters:
        x: np.ndarray[float32] - x values
        y: np.ndarray[float32] - y values
        w: np.ndarray[float32] - weights
        iper: int - periodicity flag
        span: float - smoothing span
        bass: float - bass enhancement
    
    Returns:
        np.ndarray[float32] - smoothed values
    """
    nn = x.shape[0]
    # w = np.ones(nn)
    iper = 1
    span = 0.0
    bass = 0


    
    if x.shape[0] != y.shape[0]:
        raise ValueError("Arrays x, y, and w must have the same length")
    # if x.shape[0] != y.shape[0] or x.shape[0] != w.shape[0]:
    #     raise ValueError("Arrays x, y, and w must have the same length")
    
    # Ensure arrays are contiguous
    x = np.ascontiguousarray(x, dtype=DTYPE)
    y = np.ascontiguousarray(y, dtype=DTYPE)
    # w = np.ascontiguousarray(w, dtype=DTYPE)

    # Create w array
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.ones(nn, dtype=DTYPE)
    
    # Create output array
    cdef np.ndarray[DTYPE_t, ndim=1] smo = np.empty_like(x, dtype=DTYPE)
    
    # Get the size
    cdef size_t n = x.shape[0]
    
    # Call the C function
    c_supsmu(n,
           <float*>x.data,
           <float*>y.data,
           <float*>w.data,
           iper,
           span,
           bass,
           <float*>smo.data)
    
    return smo
