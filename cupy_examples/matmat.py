import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def matmat_kernel(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    n: ct.Constant[int],
    k: ct.Constant[int],
    m: ct.Constant[int],
    tile_size_row: ct.Constant[int],
    tile_size_col: ct.Constant[int],
):
    bid_row = ct.bid(0)
    bid_col = ct.bid(1)

    a_tile = ct.load(
        a, index=(bid_row, 0), shape=(tile_size_row, k)
    )
    b_tile = ct.load(
        b, index=(0, bid_col), shape=(k, tile_size_col)
    )
    result = a_tile @ b_tile

    if bid_row * tile_size_row < n and bid_col * tile_size_col < m:
        ct.store(c, index=(bid_row, bid_col), tile=result)


def matmat(
    a: cp.ndarray,
    b: cp.ndarray,
    c: cp.ndarray,
    n: ct.Constant[int],
    k: ct.Constant[int],
    m: ct.Constant[int],
    tile_size_row: ct.Constant[int],
    tile_size_col: ct.Constant[int],
):
    assert a.shape == (n, k)
    assert b.shape == (k, m)
    assert c.shape == (n, m)

    griddims_x = ct.cdiv(n, tile_size_row)
    griddims_y = ct.cdiv(m, tile_size_col)
    grid = (griddims_x, griddims_y, 1)

    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        matmat_kernel,
        (a, b, c, n, k, m, tile_size_row, tile_size_col),
    )


def main():
    rng = cp.random.default_rng()

    n = 2**8
    k = 2**7
    m = 2**9
    tile_size = 2**4
    
    a = rng.random((n, k))
    b = rng.random((k, m))
    c = cp.zeros((n, m), dtype=a.dtype)

    matmat(a, b, c, n, k, m, tile_size, tile_size)

    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)
    expected = a_np @ b_np
    np.testing.assert_array_almost_equal(c_np, expected)

    print("✓ matmat_example passed!")


if __name__ == "__main__":
    main()
