from typing import Tuple
from teg import (
    ITeg,
    TegVar,
    Const,
    IfElse,
    Invert
)

from teg.lang.extended import (
    BiMap
)

from teg.math import (
    Cos, Sin, Sqr, ASin, Sqrt
)


from functools import partial, reduce
import operator


def teg_max(a, b):
    return IfElse(a > b, a, b)


def teg_min(a, b):
    return IfElse(a > b, b, a)


def teg_abs(a):
    return IfElse(a > 0, a, -a)


def teg_cases(exprs, conditions):
    assert len(exprs) - len(conditions) == 1, 'Need one additional expression for the default step'

    # Build ladder in reverse order.s
    exprs = exprs[::-1]
    conditions = conditions[::-1]

    if_else_ladder = exprs[0]
    for expr, condition in zip(exprs[1:], conditions):
        if_else_ladder = IfElse(condition, expr, if_else_ladder)

    return if_else_ladder


def teg_smoothstep(x):
    return IfElse(x > 0, IfElse(x < 1, 3 * Sqr(x) - 2 * Sqr(x) * x, Const(1)), Const(0))


def smoothstep(x):
    x_ = TegVar(f'{x.name}_sstep')
    inv_jacobian = (Cos(ASin(1 - 2 * x_)/3)/3) * (2/Sqrt(1 - Sqr(1 - 2 * x_)))
    return (partial(BiMap,
                    targets=[x_],
                    target_exprs=[teg_smoothstep(x)],
                    sources=[x],
                    source_exprs=[0.5 - Sin(ASin(1.0 - 2.0 * x_)/3.0)],
                    inv_jacobian=inv_jacobian,
                    target_upper_bounds=[teg_smoothstep(x.ub())],
                    target_lower_bounds=[teg_smoothstep(x.lb())]),
            x_)
