from typing import Dict, Set, Tuple, Iterable

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    ITegBool,
    Bool,
    And,
    Or,
    TegVar
)
from teg.lang.extended_utils import extract_vars
from functools import reduce
import operator


def extract_variables_from_affine(expr: ITeg) -> Dict[Tuple[str, int], ITeg]:

    if isinstance(expr, Const):
        return {}

    elif isinstance(expr, Var):
        return {(expr.name, expr.uid): expr}

    elif isinstance(expr, (Add, Mul)):
        return {**extract_variables_from_affine(expr.children[0]), **extract_variables_from_affine(expr.children[1])}

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')


def extract_moving_discontinuities(expr: ITeg,
                                   var: Var,
                                   not_ctx: Set[Tuple[str, int]],
                                   banned_variables: Set[Tuple[(str, int)]]) -> Iterable[Tuple[ITeg, ITeg]]:
    """
        Identify all subexpressions producing a moving discontinuity.
        Concretely, this means finding each branching statement including
        the integration variable and a variable defined in an outside context.
    """
    if isinstance(expr, IfElse):
        yield from moving_discontinuities_in_boolean(expr.cond, var, not_ctx, banned_variables)

    elif isinstance(expr, Teg):
        banned_variables.add((expr.dvar.name, expr.dvar.uid))

    yield from (moving_cond for child in expr.children
                for moving_cond in extract_moving_discontinuities(child, var, not_ctx, banned_variables))


def moving_discontinuities_in_boolean(expr: ITegBool,
                                      var: Var,
                                      not_ctx: Set[Tuple[str, int]],
                                      banned_variables: Set[Tuple[(str, int)]]) -> Iterable[ITeg]:
    if isinstance(expr, Bool):
        var_name_var_in_cond = extract_variables_from_affine(expr.left_expr - expr.right_expr)
        moving_var_name_uids = var_name_var_in_cond.keys() - not_ctx - {(var.name, var.uid)}

        # Check that the variable var is in the condition
        # and another variable not in not_ctx is in the condition
        if ((var.name, var.uid) in var_name_var_in_cond
                and len(moving_var_name_uids) > 0
                and len(banned_variables & moving_var_name_uids) == 0):
            yield expr

    elif isinstance(expr, (And, Or)):
        yield from moving_discontinuities_in_boolean(expr.left_expr, var, not_ctx, banned_variables)
        yield from moving_discontinuities_in_boolean(expr.right_expr, var, not_ctx, banned_variables)

    else:
        raise ValueError('Illegal expression in boolean.')


def extend_dependencies(var_list, deps_list):
    # print(var_list, 'OOO', deps_list)
    while True:
        extended_list = reduce(operator.or_, [deps_list.get(cvar, set()) for cvar in var_list])
        # print('EXTENDED_LIST: ', extended_list, ' VAR_LIST ', var_list)
        if not (extended_list - var_list):
            break
        var_list = var_list | extended_list

    return var_list | extended_list


def primitive_booleans_in(expr: ITegBool,
                          not_ctx: Set[Tuple[str, int]],
                          deps: Dict[TegVar, Set[Var]]) -> Iterable[ITeg]:

    if isinstance(expr, Bool):
        # var_name_var_in_cond = extract_variables_from_affine(expr.left_expr - expr.right_expr)
        cond_variables = extract_vars(expr.left_expr - expr.right_expr)
        extended_cond_variables = extend_dependencies(cond_variables, deps)
        moving_var_name_uids = extended_cond_variables - not_ctx

        # print(f'vnvic {var_name_var_in_cond}')
        # print(moving_var_name_uids)
        # Check that the variable var is in the condition
        # and another variable not in not_ctx is in the condition
        if (({(v.name, v.uid) for v in extended_cond_variables} & not_ctx)
            and len(moving_var_name_uids) > 0):
            yield expr

    elif isinstance(expr, (And, Or)):
        yield from primitive_booleans_in(expr.left_expr, not_ctx, deps)
        yield from primitive_booleans_in(expr.right_expr, not_ctx, deps)

    else:
        raise ValueError('Illegal expression in boolean.')