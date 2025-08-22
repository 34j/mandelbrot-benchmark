import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import taichi as ti
import torch
import typer
import warp as wp
from aquarel import load_theme
from cm_time import timer
from numba.core.errors import NumbaPerformanceWarning
from tqdm import tqdm, trange

from mandelbrot_benchmark.backends.jax import mandelbrot_jax
from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi
from mandelbrot_benchmark.backends.torch import mandelbrot_torch
from mandelbrot_benchmark.backends.warp import mandelbrot_warp

app = typer.Typer()


@app.command()
def benchmark(
    backends: str = "numba,taichi,warp",
    max_size_cpu: int = 10,
    max_size_cuda: int = 13,
    size_step: float = 0.1,
) -> None:
    """Run the Mandelbrot benchmark for different backends and devices."""
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    data = []

    # init Warp
    wp.init()
    for device in tqdm(["cpu", "cuda"], position=0, leave=False):
        # init Taichi
        if device == "cuda":
            ti.init(arch=ti.cuda, default_ip=ti.i32, default_fp=ti.f32)
        else:
            ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)
        for size in tqdm(
            (
                2
                ** np.arange(
                    1,
                    max_size_cpu if device == "cpu" else max_size_cuda,
                    step=size_step,
                )
            ).astype(int),
            position=1,
            leave=False,
        ):
            # Create a grid of complex numbers
            x, y = np.meshgrid(
                np.linspace(-2.0, 1.0, size), np.linspace(-1.5, 1.5, size)
            )
            c = x + 1j * y
            c = torch.asarray(c, dtype=torch.complex64, device=device)

            # Run each backend
            for backend in tqdm(backends.split(","), position=2, leave=False):
                for i in trange(10, position=3, leave=False):
                    with timer() as t:
                        if backend == "numba":
                            z = mandelbrot_numba(c)
                        elif backend == "taichi":
                            z = mandebrot_taichi(c)
                        elif backend == "warp":
                            z = mandelbrot_warp(c)
                        elif backend == "torch":
                            z = mandelbrot_torch(c)
                        elif backend == "jax":
                            z = mandelbrot_jax(c)
                        else:
                            raise ValueError(f"Unknown backend: {backend}")
                        str(z[0, 0].item())
                    if i == 0:
                        continue
                    data.append(
                        {
                            "backend": backend,
                            "device": device,
                            "time": t.elapsed,
                            "size": size**2,
                        }
                    )
                    torch.cuda.empty_cache()
    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)


@app.command()
def plot() -> None:
    """Plot the results."""
    theme = load_theme("boxy_dark")
    theme.apply()
    df = pd.read_csv("results.csv")
    g = sns.relplot(
        data=df,
        x="size",
        y="time",
        col="device",
        hue="backend",
        kind="line",
        style="backend",
        markers={"numba": "o", "taichi": "s", "warp": "D", "torch": "^", "jax": "v"},
    )
    g.set_xlabels("Number of pixels")
    g.set_ylabels("Time (s)")
    g.set(xscale="log", yscale="log")
    g.savefig("results.png", dpi=300)
