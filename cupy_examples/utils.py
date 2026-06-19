import numpy as np
import cupy as cp
import cuda.tile as ct
import time
from typing import Any, Sequence, TypeAlias

Dim3: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]

def assert_close(actual, desired, name: str, rtol=1e-5, atol=0.0):
    np.testing.assert_allclose(
        cp.asnumpy(actual), cp.asnumpy(desired), rtol=rtol, atol=atol
    )
    print(f"{name}: OK")


def bench(kernel, args, grid: Dim3, stream, warmup=5, iters=30):    
    for _ in range(warmup):
        ct.launch(stream, grid, kernel, args)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ct.launch(stream, grid, kernel, args)
    stream.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters