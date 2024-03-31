from typing import Dict, Set, Tuple, Iterable

from teg import ITeg, Const, Var, Add, Mul, IfElse, Teg, ITegBool, Bool, And, Or, TegVar
from teg.lang.extended_utils import extract_vars
from functools import reduce
import operator


def extract_variables_from_affine(expr: ITeg) -> Dict[Tuple[str, int], ITeg]:
    """Extract all of the variables in an affine expression."""

    if isinstance(expr, Const):
        return {}

    elif isinstance(expr, Var):
        return {(expr.name, expr.uid): expr}

    elif isinstance(expr, (Add, Mul)):
        return {**extract_variables_from_affine(expr.children[0]), **extract_variables_from_affine(expr.children[1])}

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')


def extract_moving_discontinuities(
    expr: ITeg, var: Var, not_ctx: Set[Tuple[str, int]], banned_variables: Set[Tuple[(str, int)]]
) -> Iterable[Tuple[ITeg, ITeg]]:
    """Yield all subexpressions producing a moving discontinuity.

    A moving discontinuity is a branching statement that includes
    a variable of integration and a parameter that may be differentiated.
    For example, in int_x [x < t], [x < t] is a moving discontinuity
    because x is a variable of integration and t is a variable free outside the expression.
    """
    if isinstance(expr, IfElse):
        yield from moving_discontinuities_in_boolean(expr.cond, var, not_ctx, banned_variables)

    elif isinstance(expr, Teg):
        banned_variables.add((expr.dvar.name, expr.dvar.uid))

    yield from (
        moving_cond
        for child in expr.children
        for moving_cond in extract_moving_discontinuities(child, var, not_ctx, banned_variables)
    )


def moving_discontinuities_in_boolean(
    expr: ITegBool, var: Var, not_ctx: Set[Tuple[str, int]], banned_variables: Set[Tuple[(str, int)]]
) -> Iterable[ITeg]:
    """Yield all moving discontinuities in boolean expression (e.g., in d_t int_x [x < t] )"""
    if isinstance(expr, Bool):
        var_name_var_in_cond = extract_variables_from_affine(expr.left_expr - expr.right_expr)
        moving_var_name_uids = var_name_var_in_cond.keys() - not_ctx - {(var.name, var.uid)}

        # Check that the variable var is in the condition
        # and another free variable (not in not_ctx) is in the condition
        if (
            (var.name, var.uid) in var_name_var_in_cond
            and len(moving_var_name_uids) > 0
            and len(banned_variables & moving_var_name_uids) == 0
        ):
            yield expr

    elif isinstance(expr, (And, Or)):
        yield from moving_discontinuities_in_boolean(expr.left_expr, var, not_ctx, banned_variables)
        yield from moving_discontinuities_in_boolean(expr.right_expr, var, not_ctx, banned_variables)

    else:
        raise ValueError("Illegal expression in boolean.")


def extend_dependencies(var_list, deps_list):
    while True:
        extended_list = reduce(operator.or_, [deps_list.get(cvar, set()) for cvar in var_list])
        if not (extended_list - var_list):
            break
        var_list = var_list | extended_list

    return var_list | extended_list


def primitive_booleans_in(
    expr: ITegBool, not_ctx: Set[Tuple[str, int]], deps: Dict[TegVar, Set[Var]]
) -> Iterable[ITeg]:

    if isinstance(expr, Bool):
        cond_variables = extract_vars(expr.left_expr - expr.right_expr)
        extended_cond_variables = extend_dependencies(cond_variables, deps)
        moving_var_name_uids = extended_cond_variables - not_ctx

        # Check that the variable var is in the condition
        # and another variable not in not_ctx is in the condition
        if ({(v.name, v.uid) for v in extended_cond_variables} & not_ctx) and len(moving_var_name_uids) > 0:
            yield expr

    elif isinstance(expr, (And, Or)):
        yield from primitive_booleans_in(expr.left_expr, not_ctx, deps)
        yield from primitive_booleans_in(expr.right_expr, not_ctx, deps)

    else:
        raise ValueError("Illegal expression in boolean.")
