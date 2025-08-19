from typing import Any

import numpy as np
import pytest
import taichi as ti
import torch
import warp as wp

from mandelbrot_benchmark.backends.numba import mandelbrot_numba, mandelbrot_numba_cuda
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi
from mandelbrot_benchmark.backends.warp import mandelbrot_warp


@pytest.fixture(params=["cpu", "cuda"])
def c(request: pytest.FixtureRequest) -> Any:
    device = request.param
    rng = np.random.default_rng(0)
    shape = (100, 100)
    return torch.asarray(
        rng.random(shape) + 1j * rng.random(shape), dtype=torch.complex64, device=device
    )


def test_numba(c: Any) -> None:
    if c.device.type == "cuda":
        mandelbrot_numba_cuda(c)
    else:
        mandelbrot_numba(c)


def test_taichi(c: Any) -> None:
    if c.device.type == "cuda":
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)
    mandebrot_taichi(c)


def test_warp(c: Any) -> None:
    wp.init()
    mandelbrot_warp(c)
