import numba


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
mandelbrot_numba_cuda = numba.vectorize(
    [numba.int32(numba.complex64), numba.int32(numba.complex128)], target="cuda"
)(mandelbrot)
