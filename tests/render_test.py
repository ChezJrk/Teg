
from teg import (
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup,
)
from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.reduce import reduce_to_base
from teg.maps.transform import scale, translate
from teg.maps.polar import polar_2d_map

from plot import render_image, save_image

from tap import Tap

import numpy as np


class Args(Tap):
    testname: str
    res_x: int = 64
    res_y: int = 64


args = Args().parse_args()

if args.testname == 'simple-line':
    x = TegVar('x')
    y = TegVar('y')
    t1 = Var('t1')
    t2 = Var('t2')

    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    translate_map, (x_, y_) = translate([x, y], [t1, t2])

    # Derivative of threshold only.
    integral = Teg(x_lb, x_ub,
                   Teg(y_lb, y_ub,
                       translate_map(IfElse(x_ + y_ > 0.75, 2, 1)), y
                       ), x
                   )

    integral = reduce_to_base(integral)
    image = render_image(integral, variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings={t1: 0, t2: 0}, res=(args.res_x, args.res_y))
    save_image(image, filename=f'{args.testname}.png')

elif args.testname == 'circle':
    x = TegVar('x')
    y = TegVar('y')
    r = TegVar('r')

    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    t = Var('t')

    # Area of a unit circle.
    integral = Teg(
                y_lb, y_ub,
                Teg(x_lb, x_ub,
                    polar_2d_map(IfElse(r < t, 1, 0.5), x=x, y=y, r=r), x
                    ), y
                )

    dt_expr = reduce_to_base(reverse_deriv(integral, output_list=[t])[1])
    integral = reduce_to_base(integral)
    image = render_image(integral,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings={t: 0.5},
                         bounds=((-1, 1), (-1, 1)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(image, filename=f'{args.testname}.png')

    image = render_image(dt_expr,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings={t: 0.5},
                         bounds=((-1, 1), (-1, 1)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(image, filename=f'{args.testname}_dt.png')
    pass

elif args.testname == 'rect_hyperbola':
    x = TegVar('x')
    y = TegVar('y')

    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    t = Var('t')
    t1, t2 = Var('t1'), Var('t2')
    t3, t4 = Var('t3'), Var('t4')

    scale_map, (x_s, y_s) = scale([x, y], [t1, t2])
    translate_map, (x_st, y_st) = translate([x_s, y_s], [t3, t4])

    # Area of a unit circle.
    bindings = {t1: 1, t2: 1, t3: 0, t4: 0, t: 0.25}
    # Derivative of threshold only.
    integral = Teg(
                x_lb, x_ub,
                Teg(y_lb, y_ub,
                    scale_map(translate_map(IfElse(x_st * y_st > t, 1, 0))), y
                    ), x
                )

    d_vars, dt_exprs = reverse_deriv(integral, Tup(Const(1)), output_list=[t, t1, t2, t3, t4])

    integral = reduce_to_base(integral)
    image = render_image(integral,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings=bindings,
                         bounds=((-1, 1), (-1, 1)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(np.abs(image), filename=f'{args.testname}.png')

    for d_var, dt_expr in zip(d_vars, dt_exprs):
        image = render_image(reduce_to_base(dt_expr),
                             variables=((x_lb, x_ub), (y_lb, y_ub)),
                             bindings=bindings,
                             bounds=((-1, 1), (-1, 1)),
                             res=(args.res_x, args.res_y),
                             )
        save_image(np.abs(image), filename=f'{args.testname}_{d_var.name}.png')

elif args.testname == 'rect_hyperbola_non_centered':
    x = TegVar('x')
    y = TegVar('y')

    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    t = Var('t')
    t1, t2 = Var('t1'), Var('t2')
    t3, t4 = Var('t3'), Var('t4')

    # scale_map, (x_s, y_s) = scale([x, y], [t1, t2])
    # translate_map, (x_st, y_st) = translate([x_s, y_s], [t3, t4])

    # Area of a unit circle.
    # bindings = {t1: 1, t2: 1, t3: 0, t4: 0, t: 0.25}
    bindings = {t: 0.25}
    # Derivative of threshold only.
    integral = Teg(
                x_lb, x_ub,
                Teg(y_lb, y_ub,
                    IfElse(x * y + x + y > t, 1, 0), y
                    ), x
                )

    # d_vars, dt_exprs = reverse_deriv(integral, Tup(Const(1)), output_list=[t, t1, t2, t3, t4])

    integral = reduce_to_base(integral)
    image = render_image(integral,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings=bindings,
                         bounds=((-1, 1), (-1, 1)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(np.abs(image), filename=f'{args.testname}.png')
    """
    for d_var, dt_expr in zip(d_vars, dt_exprs):
        image = render_image(reduce_to_base(dt_expr),
                             variables=((x_lb, x_ub), (y_lb, y_ub)),
                             bindings=bindings,
                             bounds=((-1, 1), (-1, 1)),
                             res=(args.res_x, args.res_y),
                             )
        save_image(np.abs(image), filename=f'{args.testname}_{d_var.name}.png')
    """
elif args.testname == 'bilinear_lerp_thresholded':
    x = TegVar('x')
    y = TegVar('y')
    t = Var('t')

    c00 = Var('c00')
    c01 = Var('c01')
    c10 = Var('c10')
    c11 = Var('c11')

    x_lb = Var('x_lb')
    x_ub = Var('x_ub')
    y_lb = Var('y_lb')
    y_ub = Var('y_ub')

    bilinear_lerp = (c00 * (1 - x) + c01 * (x)) * (1 - y) +\
                    (c10 * (1 - x) + c11 * (x)) * (y)

    lerp = Teg(x_lb, x_ub,
               Teg(y_lb, y_ub,
                   bilinear_lerp, y
                   ), x
               )

    bindings = {c00: 0.1, c11: 0.1, c01: 0.5, c10: 0.9, t: 0.55}
    image = render_image(lerp,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings=bindings,
                         bounds=((10e-3, 1 - 10e-3), (10e-3, 1 - 10e-3)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(np.abs(image), filename=f'{args.testname}_lerp.png')

    # Derivative of threshold only.
    integral = Teg(x_lb, x_ub,
                   Teg(y_lb, y_ub,
                       IfElse(bilinear_lerp > t, 1, 0), y
                       ), x
                   )
    d_vars, dt_exprs = reverse_deriv(integral, Tup(Const(1)), output_list=[t, c00, c01, c10, c11])

    image = render_image(integral,
                         variables=((x_lb, x_ub), (y_lb, y_ub)),
                         bindings=bindings,
                         bounds=((10e-3, 1 - 10e-3), (10e-3, 1 - 10e-3)),
                         res=(args.res_x, args.res_y),
                         )
    save_image(np.abs(image), filename=f'{args.testname}.png')

    for d_var, dt_expr in zip(d_vars, dt_exprs):
        image = render_image(reduce_to_base(dt_expr),
                             variables=((x_lb, x_ub), (y_lb, y_ub)),
                             bindings=bindings,
                             bounds=((10e-3, 1 - 10e-3), (10e-3, 1 - 10e-3)),
                             res=(args.res_x, args.res_y),
                             )
        save_image(np.abs(image), filename=f'{args.testname}_{d_var.name}.png')
else:
    print(f'Invalid test name: "{args.testname}"')

"""
_, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t1, t2])
# print(simplify(d_t_expr))
fd_d_t1 = 1.0  # finite_difference(reduce_to_base(integral), t1, bindings={t1: 0, t2: 0})
fd_d_t2 = 1.0  # finite_difference(reduce_to_base(integral), t2, bindings={t1: 0, t2: 0})

d_t_expr = reduce_to_base(d_t_expr)
"""