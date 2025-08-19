from typing import Any

from array_api_compat import array_namespace


# @torch.compile
def mandelbrot_torch(c: Any) -> Any:
    """Pure Python implementation of the Mandelbrot set."""
    xp = array_namespace(c)
    counter = xp.full(c.shape, 200, dtype=xp.int32, device=c.device)
    z = xp.zeros_like(c)
    for i in range(200):
        z = z * z + c
        idx = (xp.abs(z) > 2) & (counter == 200)
        counter[idx] = i
    return counter
