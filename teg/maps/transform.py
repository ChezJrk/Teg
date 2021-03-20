from typing import Tuple
from functools import partial, reduce
import operator

from teg import (
    ITeg,
    TegVar,
    Const,
    IfElse,
    Invert
)
from teg.lang.extended import BiMap
from teg.math import Cos, Sin


def translate(in_vars: Tuple[TegVar], translate: Tuple[ITeg]):
    out_vars = [TegVar(f'{in_var.name}_t') for in_var in in_vars]
    return (partial(BiMap,
                    targets=out_vars,
                    target_exprs=[in_var + t for (in_var, t) in zip(in_vars,  translate)],
                    sources=in_vars,
                    source_exprs=[out_var - t for (out_var, t) in zip(out_vars, translate)],
                    inv_jacobian=Const(1),
                    target_upper_bounds=[in_var.ub() + t for (in_var, t) in zip(in_vars, translate)],
                    target_lower_bounds=[in_var.lb() + t for (in_var, t) in zip(in_vars, translate)]),
            out_vars)


def teg_abs(x):
    return IfElse(x > 0, x, -x)


def scale(in_vars: Tuple[TegVar], scale: Tuple[ITeg]):
    out_vars = [TegVar(f'{in_var.name}_t') for in_var in in_vars]
    return (partial(BiMap,
                    targets=out_vars,
                    target_exprs=[in_var * s for (in_var, s) in zip(in_vars, scale)],
                    sources=in_vars,
                    source_exprs=[out_var / s for (out_var, s) in zip(out_vars, scale)],
                    inv_jacobian=teg_abs(Invert(reduce(operator.mul, scale))),
                    target_upper_bounds=[s * IfElse(s > 0, in_var.ub(), in_var.lb())
                                         for (in_var, s) in zip(in_vars, scale)],
                    target_lower_bounds=[s * IfElse(s > 0, in_var.lb(), in_var.ub())
                                         for (in_var, s) in zip(in_vars, scale)]),
            out_vars)


def rotate_2d(x, y, theta):
    x_ = TegVar('x_')
    y_ = TegVar('y_')
    return (partial(BiMap,
                    targets=[x_, y_],
                    target_exprs=[x * Cos(theta) + y * Sin(theta), -x * Sin(theta) + y * Cos(theta)],
                    sources=[x, y],
                    source_exprs=[x_ * Cos(theta) - y_ * Sin(theta), x_ * Sin(theta) + y_ * Cos(theta)],
                    inv_jacobian=Const(1),
                    target_lower_bounds=[Cos(theta) * IfElse(Cos(theta) > 0, x.lb(), x.ub()) +
                                         Sin(theta) * IfElse(Sin(theta) > 0, y.lb(), y.ub()),
                                         -Sin(theta) * IfElse(Sin(theta) > 0, x.ub(), x.lb()) +
                                         Cos(theta) * IfElse(Cos(theta) > 0, y.lb(), y.ub())],
                    target_upper_bounds=[Cos(theta) * IfElse(Cos(theta) > 0, x.ub(), x.lb()) +
                                         Sin(theta) * IfElse(Sin(theta) > 0, y.ub(), y.lb()),
                                         -Sin(theta) * IfElse(Sin(theta) > 0, x.lb(), x.ub()) +
                                         Cos(theta) * IfElse(Cos(theta) > 0, y.ub(), y.lb())]),
            [x_, y_])
