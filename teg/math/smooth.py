"""
    Declares popularly used smooth functions.
"""

from teg import (
    ITeg,
    Const,
    SmoothFunc,
    Invert,
)

import numpy as np


class Sqrt(SmoothFunc):
    """
        y = sqrt(x)
        TODO: Do we need bounds checks?
    """
    def __init__(self, expr: ITeg, name: str = "Sqrt"):
        super(Sqrt, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Invert(Const(2) * Sqrt(self.expr)) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Invert(Const(2) * Sqrt(self.expr))

    def operation(self, in_value):
        return np.sqrt(in_value)

    def output_size(input_size):
        return input_size


class Sqr(SmoothFunc):
    """
        y = x**2
    """
    def __init__(self, expr: ITeg, name: str = "Sqr"):
        super(Sqr, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Const(2) * self.expr * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Const(2) * self.expr

    def operation(self, in_value):
        return np.power(in_value, 2)

    def output_size(input_size):
        return input_size
