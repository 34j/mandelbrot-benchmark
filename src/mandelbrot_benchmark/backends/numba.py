from typing import Any

import numba
from numba.cuda import as_cuda_array


def mandelbrot(c: complex) -> int:
    """Pure Python implementation of the Mandelbrot set."""
    counter = 0
    z = 0j
    for _ in range(1000):
        z = z**2 + c
        if z.real**2 + z.imag**2 >= 4:
            break
        counter += 1
    return counter


mandelbrot_numba = numba.vectorize(
    [numba.int32(numba.complex64), numba.int32(numba.complex128)], target="parallel"
)(mandelbrot)
_mandelbrot_numba_cuda = numba.vectorize(
    [numba.int32(numba.complex64), numba.int32(numba.complex128)], target="cuda"
)(mandelbrot)


def mandelbrot_numba_cuda(c: Any) -> Any:
    """
    Numba implementation of the Mandelbrot set on CUDA.

    Parameters
    ----------
    c : Any
        Input array of complex numbers.

    Returns
    -------
    Any
        Output array of integers representing the Mandelbrot set.

    """
    c = as_cuda_array(c)
    return _mandelbrot_numba_cuda(c)
