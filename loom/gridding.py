import numpy


def uniform(min_val, max_val, point_count):
    grid = numpy.array(range(point_count)) + 0.5
    grid *= (max_val - min_val) / float(point_count)
    grid += min_val
    return grid


def center_heavy(min_val, max_val, point_count):
    grid = uniform(-1, 1, point_count)
    grid = numpy.arcsin(grid) / numpy.pi + 0.5
    grid *= max_val - min_val
    grid += min_val
    return grid


def left_heavy(min_val, max_val, point_count):
    grid = uniform(0, 1, point_count)
    grid = grid ** 2
    grid *= max_val - min_val
    grid += min_val
    return grid


def right_heavy(min_val, max_val, point_count):
    grid = left_heavy(max_val, min_val, point_count)
    return grid[::-1].copy()


def pitman_yor(
        min_alpha=0.1,
        max_alpha=100,
        min_d=0,
        max_d=0.5,
        alpha_count=20,
        d_count=10):
    '''
    For d = 0, this degenerates to the CRP, where the expected number of
    tables is:
        E[table_count] = O(alpha log(customer_count))
    '''
    min_alpha = float(min_alpha)
    max_alpha = float(max_alpha)
    min_d = float(min_d)
    max_d = float(max_d)

    lower_triangle = [
        (x, y)
        for x in center_heavy(0, 1, alpha_count)
        for y in left_heavy(0, 1, d_count)
        if x + y < 1
    ]
    alpha = lambda x: min_alpha * (max_alpha / min_alpha) ** x
    d = lambda y: min_d + (max_d - min_d) * y
    grid = [
        {'alpha': alpha(x), 'd': d(y)}
        for (x, y) in lower_triangle
    ]
    return grid


def bnb(min_log2_alpha=-3, max_log2_r=12):
    assert min_log2_alpha <= 0
    assert max_log2_r >= 0
    alpha_grid = [2.0 ** p for p in xrange(min_log2_alpha, 1)]
    beta_grid = [2.0 ** p for p in xrange(0, max_log2_r + 1)]
    r_grid = [2 ** p for p in xrange(0, max_log2_r + 1)]
    max_beta_r = 2 ** max_log2_r
    grid = [
        {'alpha': alpha, 'beta': beta, 'r': r}
        for r in r_grid
        for beta in beta_grid
        if beta * r <= max_beta_r
        for alpha in alpha_grid
    ]
    assert grid, 'no feasible grid points'
    return grid
