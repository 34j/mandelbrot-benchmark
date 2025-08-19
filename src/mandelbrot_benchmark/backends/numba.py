from typing import Any

import numba
from numba.cuda import as_cuda_array


def mandelbrot(c: complex) -> int:
    """Pure Python implementation of the Mandelbrot set."""
    counter = 0
    z = 0j
    for _ in range(200):
        z = z * z + c
        if z.real**2 + z.imag**2 >= 4:
            break
        counter += 1
    return counter


_mandelbrot_numba = numba.vectorize(
    [numba.int32(numba.complex64), numba.int32(numba.complex128)], target="parallel"
)(mandelbrot)
_mandelbrot_numba_cuda = numba.vectorize(
    [numba.int32(numba.complex64), numba.int32(numba.complex128)], target="cuda"
)(mandelbrot)


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
