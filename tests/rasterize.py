from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from integrable_program import (
    ITeg,
    Const,
    Var,
    IfElse,
    Teg,
    Tup,
    LetIn,
)
from teg.derivs import FwdDeriv
from teg.eval import numpy_eval as evaluate
from teg.passes.simplify import simplify
from tap import Tap


class Args(Tap):
    pixel_width: int = 10
    pixel_height: int = 10
    num_samples: int = 10


def rasterize_triangles():
    x, y = Var('x'), Var('y')
    theta = Var('theta', 0)

    def right_triangle(x0, y0):
        """ ◥ with upper right corner at (x0, y0) """
        return (y < y0) & (x < x0 + theta) & (x - x0 + y - y0 + 0.75 + theta > 0)

    inside_front_cond = right_triangle(0.7, 0.7)

    body = IfElse(inside_front_cond, 1, 0)

    pixels = []
    w, h = args.pixel_width, args.pixel_height
    for i in range(w):
        for j in range(h):
            integral = Teg(i / w, (i + 1) / w, Teg(j / h, (j + 1) / h, body, x), y)
            pixels.append(integral)
    return theta, pixels


def rasterize_triangles_no_conjunctions(args: Args) -> Tuple[ITeg, List[List[ITeg]]]:
    x, y = Var('x'), Var('y')
    theta = Var('theta', 0)

    def right_triangle(x0, y0):
        """ ◥ with upper right corner at (x0, y0) """
        below_horiz_line = IfElse(y < y0, 1, 0)
        left_of_vert_line = IfElse(x < x0, below_horiz_line, 0)
        above_diag = IfElse(x - x0 + y - y0 + 0.75 + theta > 0, left_of_vert_line, 0)
        return above_diag

    inside_back = right_triangle(1, 1)

    x0, y0 = 0.7, 0.7
    below_horiz_line = IfElse(y < y0, -1, inside_back)
    left_of_vert_line = IfElse(x < x0, below_horiz_line, inside_back)
    body = IfElse(x - x0 + y - y0 + 0.75 + theta > 0, left_of_vert_line, inside_back)

    pixels = []
    w, h = args.pixel_width, args.pixel_height
    for i in range(w):
        for j in range(h):
            integral = Teg(i / w, (i + 1) / w, Teg(j / h, (j + 1) / h, body, x), y)
            pixels.append(integral)
    return theta, pixels


if __name__ == '__main__':
    args = Args().parse_args()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    # theta, pixels = rasterize_triangles_no_conjunctions(args)
    theta, pixels = rasterize_triangles()

    w, h = args.pixel_width, args.pixel_height
    res = [evaluate(pixel, num_samples=args.num_samples) for pixel in pixels]
    pixel_grid = np.array(res).reshape((w, h))
    axes[0].imshow(pixel_grid[::-1, :], vmin=-1/(w * h), vmax=1/(w * h))

    derivs = [FwdDeriv(pixel, [(theta, 1)]) for pixel in pixels]

    res = [evaluate(deriv, num_samples=args.num_samples) for deriv in derivs]
    pixel_grid = np.array(res).reshape((w, h))
    axes[1].imshow(pixel_grid[::-1, :], vmin=-0.05, vmax=0.05)
    plt.show()
