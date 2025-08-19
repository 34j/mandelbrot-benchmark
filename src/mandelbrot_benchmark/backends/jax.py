import jax
import jax.numpy as jnp
import torch
from dlpack import asdlpack


@jax.jit
def mandelbrot_jax(c: torch.Tensor) -> torch.Tensor:
    """JAX implementation of the Mandelbrot set."""
    c = jnp.from_dlpack(asdlpack(c))
    counter = jnp.zeros(c.shape[0], dtype=jnp.int32)
    z = jnp.zeros_like(c, dtype=c.dtype)
    for i in range(20):
        z = z**2 + c
        diverged = jnp.abs(z) > 2
        counter = jnp.where(
            diverged, i, counter
        )  # use a ternary operator ("where") instead of masked-assignment
    return torch.from_dlpack(asdlpack(counter))
