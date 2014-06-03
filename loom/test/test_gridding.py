import numpy
import loom.gridding


def random_real():
    return numpy.arctan(numpy.pi * (numpy.random.random() - 0.5))


def _test_bounds(make_grid):
    ITERS = 100
    for i in range(ITERS):
        x = random_real()
        y = random_real()
        if y < x:
            x, y = y, x
        grid = make_grid(x, y, 100)
        assert x <= grid.min()
        assert grid.max() <= y


def test_uniform():
    _test_bounds(loom.gridding.uniform)


def test_center_heavy():
    _test_bounds(loom.gridding.center_heavy)


def test_left_heavy():
    _test_bounds(loom.gridding.left_heavy)
