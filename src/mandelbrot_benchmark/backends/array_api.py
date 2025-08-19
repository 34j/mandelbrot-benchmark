from typing import Any

from array_api_compat import array_namespace


def mandelbrot_vectorized(c: Any) -> Any:
    """Pure Python implementation of the Mandelbrot set."""
    xp = array_namespace(c)
    counter = xp.zeros(c.shape, dtype=xp.int32, device=c.device)
    z = xp.zeros_like(c)
    for _ in range(1000):
        z = z**2 + c
        idx = z.real**2 + z.imag**2 < 4
        if not xp.any(idx):
            break
        counter[idx] += 1
    return counter
