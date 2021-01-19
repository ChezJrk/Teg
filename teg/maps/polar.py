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

import numpy as np


def teg_max(a, b):
    return IfElse(a > b, a, b)


def teg_min(a, b):
    return IfElse(a > b, b, a)


TEG_NEGATIVE_PI = Const(-np.pi)
TEG_PI = Const(np.pi)
TEG_2_PI = Const(2 * np.pi)


def polar_2d_map(expr, x, y, r):
    """
    Create a polar 2D map with x=0, y=0 as center and negative y axis as 0 & 2PI
    """
    theta = TegVar('theta')

    distance_to_origin = Sqrt(Sqr((y.lb() + y.ub()) / 2) + Sqr((x.lb() + x.ub()) / 2))
    box_radius = Sqrt(Sqr((y.ub() - y.lb()) / 2) + Sqr((x.ub() - x.lb()) / 2))

    """
        Manual interval arithmetic for conservative polar bounds.
        (These are not strictly necessary.
         Using (0,2pi) still produces the correct unbiased integrals. However,
         they will have terrible sample behaviour)
    """
    box_upper_right = ATan2(Tup(x.ub(), y.ub()))
    box_lower_right = ATan2(Tup(x.ub(), y.lb()))
    box_upper_left = ATan2(Tup(x.lb(), y.ub()))
    box_lower_left = ATan2(Tup(x.lb(), y.lb()))

    right_theta_lower = IfElse(y.ub() > 0,
                               IfElse(x.lb() > 0,
                                      box_upper_left,
                                      0
                                      ),
                               box_upper_right
                               )
    right_theta_upper = IfElse(y.lb() < 0,
                               IfElse(x.lb() > 0,
                                      box_lower_left,
                                      TEG_PI
                                      ),
                               box_lower_right
                               )

    left_theta_upper = IfElse(y.ub() > 0,
                              IfElse(x.ub() < 0,
                                     box_upper_right,
                                     0
                                     ),
                              box_upper_left
                              )
    left_theta_lower = IfElse(y.lb() < 0,
                              IfElse(x.ub() < 0,
                                     box_lower_right,
                                     TEG_NEGATIVE_PI
                                     ),
                              box_lower_left
                              )

    return IfElse(
                x > 0,
                BiMap(expr,
                      sources=[x, y],
                      source_exprs=[r * Sin(theta), r * Cos(theta)],
                      targets=[r, theta],
                      target_exprs=[Sqrt(Sqr(x) + Sqr(y)), ATan2(Tup(x, y))],
                      inv_jacobian=r,
                      target_lower_bounds=[
                          teg_max(distance_to_origin - box_radius, 0),
                          right_theta_lower
                      ],
                      target_upper_bounds=[
                          distance_to_origin + box_radius,
                          right_theta_upper
                      ]
                      ),
                BiMap(expr,
                      sources=[x, y],
                      source_exprs=[r * Sin(theta), r * Cos(theta)],
                      targets=[r, theta],
                      target_exprs=[Sqrt(Sqr(x) + Sqr(y)), ATan2(Tup(x, y))],
                      inv_jacobian=r,
                      target_lower_bounds=[
                        teg_max(distance_to_origin - box_radius, 0),
                        left_theta_lower
                      ],
                      target_upper_bounds=[
                        distance_to_origin + box_radius,
                        left_theta_upper
                      ]
                      )
                )
