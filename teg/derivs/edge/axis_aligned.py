from typing import Dict, Set, List, Tuple
from functools import reduce

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    true,
    false,
)

from teg.passes.substitute import substitute
from .common import extract_moving_discontinuities


def extract_constants_from_affine(expr: ITeg) -> List[Const]:

    if isinstance(expr, Const):
        return [expr]

    elif isinstance(expr, Var):
        return []

    elif isinstance(expr, (Add, Mul)):
        return extract_constants_from_affine(expr.children[0]) + extract_constants_from_affine(expr.children[1])

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')


def extract_variables_from_affine(expr: ITeg) -> Dict[Tuple[str, int], Var]:

    if isinstance(expr, Const):
        return {}

    elif isinstance(expr, Var):
        return {(expr.name, expr.uid): expr}

    elif isinstance(expr, (Add, Mul)):
        return {**extract_variables_from_affine(expr.children[0]), **extract_variables_from_affine(expr.children[1])}

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')


def solve_for_dvar(expr: ITeg, var: ITeg):

    def flatten_to_nary_add(expr: ITeg) -> List[ITeg]:

        if isinstance(expr, Var):
            return [expr]

        elif isinstance(expr, Mul):
            if expr.children[0] == Const(-1):
                return [-e for e in flatten_to_nary_add(expr.children[1])]
            else:
                return [expr]

        elif isinstance(expr, Add):
            return [e for ex in expr.children for e in flatten_to_nary_add(ex)]

        else:
            raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')

    def prod(consts: List[Const]) -> ITeg:
        # NOTE: Butchering constants...
        return reduce(lambda x, y: x * y, [c.value for c in consts], 1)

    # Combine all common terms in the constant vector pairs
    d, const = {}, 0
    for e in flatten_to_nary_add(expr):
        vs = extract_variables_from_affine(e).values()
        cs = extract_constants_from_affine(e)
        assert len(vs) <= 1, "Only a single variable can be in an affine product"
        if len(vs) == 1:
            v = list(vs)[0]
            if v.name in d:
                d[(v.name, v.uid)] = (d[(v.name, v.uid)][0] + cs, v)
            else:
                d[(v.name, v.uid)] = cs, v
        elif len(cs) > 0:
            const += prod(cs)

    # Remove var.name and negate all constants
    cks, vs = d.pop((var.name, var.uid))
    ck_val = prod(cks)

    # Aggregate all of the constants and variables
    inverse = -const / ck_val
    for cs, v in d.values():
        inverse += Const(-prod(cs) / ck_val) * v
    return inverse


def delta_contribution(expr: Teg,
                       not_ctx: Set[Tuple[str, int]]
                       ) -> Dict[Tuple[str, int], Tuple[Tuple[str, int], ITeg]]:
    """Given an expression for the integral, generate an expression for the derivative of jump discontinuities. """

    # Descends into all subexpressions extracting moving discontinuities
    moving_var_data, considered_bools = [], []
    for discont_bool in extract_moving_discontinuities(expr.body, expr.dvar, not_ctx.copy(), set()):
        try:
            considered_bools.index(discont_bool)
        except ValueError:
            considered_bools.append(discont_bool)

            # Evaluate the discontinuity at x = t+
            expr_body_right = expr.body
            expr_body_right = substitute(expr_body_right, discont_bool, false)

            expr_for_dvar = solve_for_dvar(discont_bool.left_expr - discont_bool.right_expr, expr.dvar)
            expr_body_right = substitute(expr_body_right, expr.dvar, expr_for_dvar)

            # Evaluate the discontinuity at x = t-
            expr_body_left = expr.body
            expr_body_left = substitute(expr_body_left, discont_bool, true)
            expr_body_left = substitute(expr_body_left, expr.dvar, expr_for_dvar)

            # if lower < dvar < upper, include the contribution from the discontinuity (x=t+ - x=t-)
            discontinuity_happens = (expr.lower < expr_for_dvar) & (expr_for_dvar < expr.upper)
            moving_var_delta = IfElse(discontinuity_happens, expr_body_left - expr_body_right, Const(0))
            moving_var_data.append((moving_var_delta, expr_for_dvar))

    return moving_var_data
