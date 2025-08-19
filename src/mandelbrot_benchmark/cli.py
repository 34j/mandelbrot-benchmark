import numpy as np
import pandas as pd
import seaborn as sns
import taichi as ti
import torch
import typer
import warp as wp
from cm_time import timer

from mandelbrot_benchmark.backends.jax import mandelbrot_jax
from mandelbrot_benchmark.backends.numba import mandelbrot_numba
from mandelbrot_benchmark.backends.taichi import mandebrot_taichi
from mandelbrot_benchmark.backends.torch import mandelbrot_torch
from mandelbrot_benchmark.backends.warp import mandelbrot_warp

app = typer.Typer()


@app.command()
def benchmark() -> None:
    """Run the Mandelbrot benchmark for different backends and devices."""
    data = []

    # init Warp
    wp.init()
    for device in ["cpu", "cuda"]:
        # init Taichi
        if device == "cuda":
            ti.init(arch=ti.cuda, default_ip=ti.i32, default_fp=ti.f32)
        else:
            ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)
        for size in 2 ** np.arange(1, 12):
            # Create a grid of complex numbers
            x, y = np.meshgrid(
                np.linspace(-2.0, 1.0, size), np.linspace(-1.5, 1.5, size)
            )
            c = x + 1j * y
            c = torch.asarray(c, dtype=torch.complex64, device=device)

            # Run each backend
            for backend in ["numba", "taichi", "warp", "torch", "jax"]:
                for i in range(10):
                    with timer() as t:
                        if backend == "numba":
                            mandelbrot_numba(c)
                        elif backend == "taichi":
                            mandebrot_taichi(c)
                        elif backend == "warp":
                            mandelbrot_warp(c)
                        elif backend == "torch":
                            mandelbrot_torch(c)
                        elif backend == "jax":
                            mandelbrot_jax(c)
                        else:
                            raise ValueError(f"Unknown backend: {backend}")
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
    df = pd.DataFrame(data)
    df.to_csv("results.csv", index=False)


@app.command()
def plot() -> None:
    """Plot the results."""
    df = pd.read_csv("results.csv")
    g = sns.relplot(
        data=df,
        x="size",
        y="time",
        row="device",
        hue="backend",
        kind="line",
        style="backend",
        markers={"numba": "o", "taichi": "s", "warp": "D", "torch": "^", "jax": "v"},
    )
    g.set_xlabels("Number of pixels")
    g.set_ylabels("Time (s)")
    g.set(xscale="log", yscale="log")
    g.savefig("results.png", dpi=300)
