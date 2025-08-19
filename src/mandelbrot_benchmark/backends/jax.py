import jax
import jax.numpy as jnp
import torch
from dlpack import asdlpack


@jax.jit
def _mandelbrot_jax(c: jnp.ndarray) -> jnp.ndarray:
    """JAX implementation of the Mandelbrot set."""
    counter = jnp.full(c.shape[0], -1, dtype=jnp.int32)
    z = jnp.zeros_like(c, dtype=c.dtype)
    for i in range(200):
        z = z**2 + c
        counter = jnp.where(
            jnp.abs(z) > 2 & (counter == -1), i, counter
        )  # use a ternary operator ("where") instead of masked-assignment
    return counter


def mandelbrot_jax(c: torch.Tensor) -> torch.Tensor:
    """
    JAX implementation of the Mandelbrot set.

    Parameters
    ----------
    c : torch.Tensor
        Input array of complex numbers.

    Returns
    -------
    torch.Tensor
        Output array of integers representing the Mandelbrot set.

    """
    # https://github.com/jax-ml/jax/issues/1100
    c = jnp.from_dlpack(asdlpack(c))
    out = _mandelbrot_jax(c)
    return torch.from_dlpack(asdlpack(out))
