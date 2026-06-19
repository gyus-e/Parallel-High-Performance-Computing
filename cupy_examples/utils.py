import numpy as np
import cupy as cp
import cuda.tile as ct
import time
from typing import TypeAlias

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


def swizzle_2d(n: int, m: int, tn: int, tm: int, group_size_m: int) -> tuple[int, int]:
    pid = ct.bid(0)
    num_tiles_n = ct.cdiv(n, tn)
    num_tiles_m = ct.cdiv(m, tm)

    num_tiles_in_group = group_size_m * num_tiles_n
    group_id = pid // num_tiles_in_group
    first_tile_m = group_id * group_size_m

    diff = num_tiles_m - first_tile_m
    group_size_m = min(diff, group_size_m)
    pid_in_group = pid % num_tiles_in_group

    bidx = first_tile_m + (pid_in_group % group_size_m)
    bidy = pid_in_group // group_size_m

    return bidx, bidy
