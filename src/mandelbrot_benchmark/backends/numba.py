from typing import Any

import numba
from numba.cuda import as_cuda_array
from numba.cuda.cudadrv.error import CudaSupportError


def _mandelbrot_32(c: Any) -> Any:
    """Pure Python implementation of the Mandelbrot set."""
    counter = numba.int32(200)
    z = numba.complex64(0)
    for i in range(200):
        z = z * z + c
        if z.real**2 + z.imag**2 >= 4:
            counter = i
            break
    return counter


_mandelbrot_numba = numba.vectorize(
    [numba.int32(numba.complex64)], target="parallel", nopython=True, fastmath=True
)(_mandelbrot_32)
try:
    _mandelbrot_numba_cuda = numba.vectorize(
        [numba.int32(numba.complex64)],
        target="cuda",
    )(_mandelbrot_32)
except CudaSupportError:
    _mandelbrot_numba_cuda = None


def mandelbrot_numba(c: Any) -> Any:
    """
    Numba implementation of the Mandelbrot set.

    Parameters
    ----------
    c : Any
        Input array of complex numbers.

    Returns
    -------
    Any
        Output array of integers representing the Mandelbrot set.

    """
    if "cuda" in str(c.device):
        c = as_cuda_array(c)
        return _mandelbrot_numba_cuda(c)
    else:
        return _mandelbrot_numba(c)
