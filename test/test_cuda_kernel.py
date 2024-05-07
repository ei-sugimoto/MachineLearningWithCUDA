import math

import numpy as np
from numba import cuda

a = np.arange(8192, dtype=np.float32)
a = np.full(8192, 3, dtype=np.float32)
out = np.empty_like(a)


@cuda.jit
def gpu_sqrt_kernel(x: np.ndarray, out: np.ndarray):
    idx = cuda.grid(1)  # type: ignore
    if idx < x.size:  # Avoid out-of-bounds access
        out[idx] = math.sqrt(x[idx])


def test_gpu_sqrt():
    threads_per_block = 256
    blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block
    gpu_sqrt_kernel[blocks_per_grid, threads_per_block](a, out)  # type: ignore
    assert np.allclose(out, np.sqrt(a))
