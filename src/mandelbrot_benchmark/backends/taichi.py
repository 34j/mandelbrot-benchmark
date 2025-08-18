from typing import Any

import taichi as ti
import taichi.math as tm
from array_api_compat import array_namespace


@ti.func
def mandelbrot_func(c: tm.vec2) -> ti.i32:
    counter = 0
    z = tm.vec2(0.0, 0.0)
    for i in range(1000):
        z = tm.vec2(z.x**2 - z.y**2, 2 * z.x * z.y) + c
        if z.x**2 + z.y**2 >= 4:
            break
        counter += 1
    return counter


@ti.kernel
def mandelbrot_kernel(field: ti.template(), out: ti.template()):
    for i, j in field:
        out[i, j] = mandelbrot_func(field[i, j])


def mandebrot_taichi(c: Any) -> Any:
    xp = array_namespace(c)
    out = xp.empty(c.shape, dtype=xp.int32, device=c.device)
    mandelbrot_kernel(c, out)
    return out
