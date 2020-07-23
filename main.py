import numpy as np
import matplotlib.pyplot as plt

from integrable_program import (
    ITeg,
    Const,
    Var,
    Cond,
    Teg,
    Tup,
    LetIn,
)
from derivs import FwdDeriv
from evaluate import evaluate
from simplify import simplify


def rasterize_triangles():
    x, y = Var('x'), Var('y')
    theta = Var('theta', 0)

    def right_triangle(x0, y0):
        """ ◥ with upper right corner at (x0, y0) """
        below_horiz_line = Cond(y - y0, 1, 0)
        left_of_vert_line = Cond(x - x0, below_horiz_line, 0)
        above_diag = Cond(x - x0 + y - y0 + 0.75 + theta, 0, left_of_vert_line)
        return above_diag

    # inside_back = right_triangle(1, 1)
    inside_front = right_triangle(0.95, 0.95)

    # body = Cond(inside_front, 1, inside_back)

    pixels = []
    w, h = 30, 30
    for i in range(w):
        for j in range(h):
            integral = Teg(i / w, (i + 1) / w, Teg(j / h, (j + 1) / h, inside_front, x), y)
            pixels.append(integral)
    return w, h, theta, pixels


if __name__ == '__main__':

    fig, axes = plt.subplots(nrows=1, ncols=2)
    w, h, theta, pixels = rasterize_triangles()
    num_samples = 20

    res = [evaluate(pixel, num_samples=num_samples) for pixel in pixels]
    pixel_grid = np.array(res).reshape((w, h))
    axes[0].imshow(pixel_grid[::-1, :])

    derivs = [FwdDeriv(pixel, [(theta, 1)]) for pixel in pixels]
    res = [evaluate(deriv, num_samples=num_samples) for deriv in derivs]
    pixel_grid = np.array(res).reshape((w, h))
    axes[1].imshow(pixel_grid[::-1, :], vmin=-0.05, vmax=0.05)
    plt.show()
