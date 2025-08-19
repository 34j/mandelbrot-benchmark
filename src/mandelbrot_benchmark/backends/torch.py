from typing import Any

import torch
from array_api_compat import array_namespace


@torch.compile
def mandelbrot_vectorized(c: Any) -> Any:
    """Pure Python implementation of the Mandelbrot set."""
    xp = array_namespace(c)
    counter = xp.zeros(c.shape, dtype=xp.int32, device=c.device)
    z = xp.zeros_like(c)
    for _ in range(20):
        z = z * z + c
        idx = z.real**2 + z.imag**2 < 4
        if not xp.any(idx):
            break
        counter[idx] += 1
    return counter
