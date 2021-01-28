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


class Sin(SmoothFunc):
    """
        y = sin(x)
    """
    def __init__(self, expr: ITeg, name: str = "Sin"):
        super(Sin, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return Cos(self.expr) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * Cos(self.expr)

    def operation(self, in_value):
        return np.sin(in_value)

    def output_size(input_size):
        return input_size


class Cos(SmoothFunc):
    """
        y = cos(x)
    """
    def __init__(self, expr: ITeg, name: str = "Cos"):
        super(Cos, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        return -Sin(self.expr) * in_deriv_expr

    def rev_deriv(self, out_deriv_expr: ITeg):
        return out_deriv_expr * -Sin(self.expr)

    def operation(self, in_value):
        return np.cos(in_value)

    def output_size(input_size):
        return input_size


class ASin(SmoothFunc):
    """
        theta = asin(x)
    """
    def __init__(self, expr: ITeg, name: str = "ASin"):
        super(ASin, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        raise NotImplementedError

    def rev_deriv(self, out_deriv_expr: ITeg):
        raise NotImplementedError

    def operation(self, in_value):
        return np.arcsin(in_value)

    def output_size(input_size):
        return input_size


class ATan2(SmoothFunc):
    """
        theta = atan2(x, y)
    """
    def __init__(self, expr: ITeg, name: str = "ATan2"):
        super(ATan2, self).__init__(expr=expr, name=name)

    def fwd_deriv(self, in_deriv_expr: ITeg):
        raise NotImplementedError

    def rev_deriv(self, out_deriv_expr: ITeg):
        raise NotImplementedError

    def operation(self, in_value):
        return np.arctan2(in_value[0], in_value[1])

    def output_size(input_size):
        assert input_size == 2
        return 1
