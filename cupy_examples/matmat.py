import cupy as cp
import cuda.tile as ct
import utils


@ct.kernel
def matmat_kernel(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    tm: ct.Constant[int],
):
    ii = ct.bid(0)
    jj = ct.bid(1)

    num_tiles = ct.num_tiles(a, axis=1, shape=(tn, tk))
    # num_tiles = ct.num_tiles(b, axis=0, shape=(tk, tm))

    acc = ct.full((tn, tm), 0.0, dtype=a.dtype)

    for kk in range(num_tiles):
        a_tile = ct.load(a, index=(ii, kk), shape=(tn, tk))
        b_tile = ct.load(b, index=(kk, jj), shape=(tk, tm))
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(c, index=(ii, jj), tile=acc)


@ct.kernel
def matmat_swizzled_kernel(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    tm: ct.Constant[int],
):
    ii, jj = utils.swizzle_2d(m=c.shape[0], n=c.shape[1], tm=tm, tn=tn, group_size_m=4)

    num_tiles = ct.num_tiles(a, axis=1, shape=(tn, tk))
    # num_tiles = ct.num_tiles(b, axis=0, shape=(tk, tm))

    acc = ct.full((tn, tm), 0.0, dtype=a.dtype)

    for kk in range(num_tiles):
        a_tile = ct.load(a, index=(ii, kk), shape=(tn, tk))
        b_tile = ct.load(b, index=(kk, jj), shape=(tk, tm))
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(c, index=(ii, jj), tile=acc)


def test_matmat(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    tm: ct.Constant[int],
    grid: utils.Dim3,
    stream: cp.cuda.Stream = cp.cuda.get_current_stream(),
):
    assert a.dtype == b.dtype == c.dtype
    assert a.shape[1] == b.shape[0]
    assert c.shape[0] == a.shape[0] and c.shape[1] == b.shape[1]

    ct.launch(
        stream,
        grid,
        matmat_kernel,
        (a, b, c, tn, tk, tm),
    )

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)
    expected = a_np @ b_np

    utils.assert_close(c_np, expected, "matmat_kernel")


def test_matmat_swizzled(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    tn: ct.Constant[int],
    tk: ct.Constant[int],
    tm: ct.Constant[int],
    grid: utils.Dim3,
    stream: cp.cuda.Stream = cp.cuda.get_current_stream(),
):
    assert a.dtype == b.dtype == c.dtype
    assert a.shape[1] == b.shape[0]
    assert c.shape[0] == a.shape[0] and c.shape[1] == b.shape[1]

    ct.launch(
        stream,
        grid,
        matmat_swizzled_kernel,
        (a, b, c, tn, tk, tm),
    )

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)
    expected = a_np @ b_np

    utils.assert_close(c_np, expected, "matmat_swizzled_kernel")


def main():
    tn = 256
    tk = 64
    tm = 128

    n = 2**10
    k = 2**8
    m = 2**9

    grid = (int(ct.cdiv(n, tn)), int(ct.cdiv(m, tm)), 1)
    stream = cp.cuda.get_current_stream()

    rng: cp.random._generator_api = cp.random.default_rng()

    a = rng.random((n, k))
    b = rng.random((k, m))
    c = cp.zeros((n, m), dtype=a.dtype)

    # Test
    test_matmat(a, b, c, tn, tk, tm, grid, stream)
    print("✓ matmat_kernel passed!")

    test_matmat_swizzled(a, b, c, tn, tk, tm, grid, stream)
    print("✓ matmat_swizzled_kernel passed!")

    # Benchmark
    time = utils.bench(
        matmat_kernel,
        (a, b, c, tn, tk, tm),
        grid=grid,
        stream=stream,
        warmup=10,
        iters=100,
    )
    print(f"matmat_kernel: {time:.3e} sec/iter")

    time = utils.bench(
        matmat_swizzled_kernel,
        (a, b, c, tn, tk, tm),
        grid=grid,
        stream=stream,
        warmup=10,
        iters=100,
    )
    print(f"matmat_swizzled_kernel: {time:.3e} sec/iter")

if __name__ == "__main__":
    main()
