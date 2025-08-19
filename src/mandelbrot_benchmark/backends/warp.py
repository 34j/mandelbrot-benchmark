import torch
import warp as wp
from array_api_compat import array_namespace

wp.init()


@wp.func
def _mandelbrot_func(c: wp.vec2) -> wp.int32:
    counter = wp.int32(0)
    z = wp.vec2(0.0, 0.0)
    for _ in range(1000):
        z = wp.vec2(z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]) + c
        if z[0] * z[0] + z[1] * z[1] >= 4.0:
            break
        counter += 1
    return counter


@wp.kernel
def _mandelbrot_kernel(
    c: wp.array2d(dtype=wp.vec2),  # type: ignore
    out: wp.array2d(dtype=wp.int32),  # type: ignore
) -> None:
    i, j = wp.tid()
    out[i, j] = _mandelbrot_func(c[i, j])


def mandelbrot_warp(c: torch.Tensor) -> torch.Tensor:
    """
    Warp implementation of the Mandelbrot set.

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
    field = wp.array(c, dtype=wp.vec2, device=device)
    wp.launch(
        kernel=_mandelbrot_kernel,
        dim=c.shape,
        inputs=[field],
        outputs=[out],
        device=device,
    )
    return wp.to_torch(out, requires_grad=False)
