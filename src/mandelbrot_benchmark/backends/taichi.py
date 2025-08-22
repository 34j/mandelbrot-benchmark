from typing import Any

import taichi as ti
import taichi.math as tm
from array_api_compat import array_namespace


@ti.func
def _mandelbrot_func(c: tm.vec2) -> ti.i32:
    counter = ti.i32(200)
    z = tm.vec2(0, 0)
    for i in range(200):
        z = tm.cpow(z, 2) + c
        if z.x**2 + z.y**2 >= 4.0:
            counter = i
            break
    return counter


@ti.kernel
def _mandelbrot_kernel(c: ti.types.ndarray(), out: ti.types.ndarray()):  # type: ignore
    for I in ti.grouped(out):
        out[I] = _mandelbrot_func(tm.vec2(c[I, 0], c[I, 1]))


def mandebrot_taichi(c: Any) -> Any:
    """
    Taichi implementation of the Mandelbrot set.

    Since Taichi does not support complex numbers directly,
    internally the input is stacked as a +1D array with real and imaginary parts.

    Taichi's from_numpy() and to_numpy() are !!NOT!! zero-copy,
    so we pass non-Taichi arrays directly to the kernel.
    (See "Note" in https://docs.taichi-lang.org/docs/external)

    See Also
    --------
    https://docs.taichi-lang.org/docs/external

    """
    xp = array_namespace(c)
    out = xp.empty(c.shape, dtype=xp.int32, device=c.device)
    c = xp.stack([c.real, c.imag], axis=-1)
    _mandelbrot_kernel(c, out)
    return out
