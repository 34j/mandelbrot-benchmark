from typing import Any

import torch
import warp as wp
from array_api_compat import array_namespace


@wp.func
def _mandelbrot_func(c: wp.vec2f) -> wp.int32:
    counter = wp.int32(200)
    z = type(c)()
    for i in range(200):
        z = wp.vec2f(z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]) + c
        if z[0] * z[0] + z[1] * z[1] >= 4.0:
            counter = i
            break
    return counter


@wp.func  # type: ignore
def _mandelbrot_func(c: wp.vec2d) -> wp.int32:
    counter = wp.int32(200)
    z = type(c)()
    for i in range(200):
        z = wp.vec2d(z[0] * z[0] - z[1] * z[1], wp.float64(2.0) * z[0] * z[1]) + c
        if z[0] * z[0] + z[1] * z[1] >= wp.float64(4.0):
            counter = i
            break
    return counter


@wp.kernel
def _mandelbrot_kernel(
    c: wp.array2d(dtype=Any),  # type: ignore
    out: wp.array2d(dtype=wp.int32),  # type: ignore
) -> None:
    i, j = wp.tid()
    out[i, j] = _mandelbrot_func(c[i, j])


@wp.overload  # type: ignore
def _mandelbrot_kernel(
    c: wp.array2d(dtype=wp.vec2f),  # type: ignore
    out: wp.array2d(dtype=wp.int32),  # type: ignore
) -> None: ...
@wp.overload  # type: ignore
def _mandelbrot_kernel(
    c: wp.array2d(dtype=wp.vec2d),  # type: ignore
    out: wp.array2d(dtype=wp.int32),  # type: ignore
) -> None: ...


def mandelbrot_warp(c: torch.Tensor) -> torch.Tensor:
    """
    Warp implementation of the Mandelbrot set.

    Warp converts external arrays to its own format zero-copy,
    therefore we do not need to worry about it.

    See Also
    --------
    https://nvidia.github.io/warp/modules/interoperability.html

    """
    if "cuda" in str(c.device):
        device = "cuda"
    else:
        device = "cpu"
    xp = array_namespace(c)
    out = wp.empty(shape=c.shape, dtype=wp.int32, device=device)
    c = xp.stack([c.real, c.imag], axis=-1)
    field = wp.array(
        c, dtype=wp.vec2d if c.dtype == torch.float64 else wp.vec2f, device=device
    )
    wp.launch(
        kernel=_mandelbrot_kernel,
        dim=c.shape,
        inputs=[field],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out, requires_grad=False)
