from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import taichi as ti
import torch
import warp as wp
from aquarel import load_theme
from array_api_compat import to_device
from numpy.testing import assert_array_equal

from mandelbrot_benchmark.backends.jax import mandelbrot_jax
from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi
from mandelbrot_benchmark.backends.torch import mandelbrot_torch
from mandelbrot_benchmark.backends.warp import mandelbrot_warp
from mandelbrot_benchmark.plot import plot_mandelbrot

wp.init()


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    Path("tests/.cache").mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def extent() -> tuple[float, float, float, float]:
    """Fixture for the extent of the Mandelbrot set."""
    return (-2.0, 1.0, -1.5, 1.5)


@pytest.fixture(params=["cpu", "cuda"])
def c(request: pytest.FixtureRequest, extent: tuple[float, float, float, float]) -> Any:
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    x, y = np.meshgrid(
        np.linspace(*extent[:2], 100),
        np.linspace(*extent[2:], 100),
    )
    c = x + 1j * y
    return torch.asarray(
        c,
        dtype=torch.complex64,
        device=device,
    )


@pytest.mark.parametrize("backend", ["numba", "taichi", "warp", "torch", "jax"])
def test_each(c: Any, backend: str, extent: tuple[float, float, float, float]) -> None:
    if backend == "numba":
        z = mandelbrot_numba(c)
    elif backend == "taichi":
        if c.device.type == "cuda":
            ti.init(arch=ti.cuda, default_ip=ti.i32, default_fp=ti.f32)
        else:
            ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)
        z = mandebrot_taichi(c)
    elif backend == "warp":
        z = mandelbrot_warp(c)
    elif backend == "torch":
        z = mandelbrot_torch(c)
    elif backend == "jax":
        z = mandelbrot_jax(c)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    theme = load_theme("boxy_dark")
    theme.apply()
    fig, ax = plt.subplots()
    if backend != "numba":
        z = to_device(z, "cpu")
    plot_mandelbrot(z, ax=ax, extent=extent)
    fig.savefig(f"tests/.cache/test_backends_{backend}_{c.device.type}.png")


def test_all_same(c: Any) -> None:
    """Test that all backends return the same result."""
    pytest.skip("Unstable")
    if c.device.type == "cuda":
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)
    wp.init()

    numba_result = mandelbrot_numba(c)
    taichi_result = mandebrot_taichi(c)
    warp_result = mandelbrot_warp(c)
    vectorized_result = mandelbrot_torch(c)

    assert_array_equal(numba_result, warp_result)
    assert_array_equal(numba_result, taichi_result)
    assert_array_equal(numba_result, vectorized_result)
