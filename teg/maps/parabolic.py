from teg import (
    TegVar,
    IfElse,
    Tup,
    Const
)

from teg.lang.extended import (
    BiMap
)

from teg.math import (
    Sin,
    Cos,
    Sqrt,
    ATan2,
    Sqr
)

from teg.derivs.jacobian import fill_jacobian
import numpy as np


def teg_max(a, b):
    return IfElse(a > b, a, b)


def teg_min(a, b):
    return IfElse(a > b, b, a)


def parabolic_map(expr, x, x_):
    """
    Creates a 2-stage parabolic map for x -> x^2
    """

    bimap_left = BiMap(expr,
                       sources=[x],
                       source_exprs=[Sqrt(x_)],
                       targets=[x_],
                       target_exprs=[Sqr(x)],  # inverses
                       inv_jacobian=None,
                       target_lower_bounds=[
                           Sqrt(teg_max(x.lb(), 0))
                       ],
                       target_upper_bounds=[
                           Sqrt(teg_max(x.ub(), 0))
                       ]
                       )

    bimap_right = BiMap(expr,
                        sources=[x],
                        source_exprs=[Sqrt(x_)],
                        targets=[x_],
                        target_exprs=[Sqr(x)],  # inverses
                        inv_jacobian=None,
                        target_lower_bounds=[
                            Sqrt(teg_max(x.lb(), 0))
                        ],
                        target_upper_bounds=[
                            Sqrt(teg_max(x.ub(), 0))
                        ]
                        )

    return IfElse(x > 0,
                  fill_jacobian(bimap_left),
                  fill_jacobian(bimap_right)
                  )
