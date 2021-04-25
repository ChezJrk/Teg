from typing import Dict
from functools import reduce
import operator

from teg import (
    ITeg,
    Const,
    Var,
    TegVar,
    IfElse,
    LetIn,
)
from teg.lang.extended import ITegExtended, BiMap, Delta
from teg.lang.markers import Placeholder
from teg.passes.substitute import substitute
from teg.derivs import fwd_deriv


def is_base_language(expr: ITeg):
    """Checks if the tree has any elements from the extended language.

    Trees cannot be evaluated in the extended language and must first be reduced to the base language.
    """
    if isinstance(expr, ITegExtended):
        return False
    elif hasattr(expr, 'children'):
        return all([is_base_language(child) for child in expr.children])
    else:
        return True


def contains_delta(expr: ITeg):
    """Checks if an expression contains a Dirac delta"""
    if isinstance(expr, Delta):
        return True
    elif hasattr(expr, 'children'):
        return any([contains_delta(child) for child in expr.children])
    else:
        return False


def is_delta_normal(expr: Delta):
    """Check is if the expression in a Dirac delta function is just a variable."""
    if isinstance(expr.expr, TegVar):
        return True
    else:
        return False


def is_bimap_trivial(expr: BiMap):
    """Checks if the BiMap has no Delta element in its body.

    A BiMap without any delta elements can be converted into a LetIn expression without any consequence.
    """
    return not contains_delta(expr.expr)


def find_in(expr: ITeg, fn):
    if fn(expr):
        return expr

    exprs = [ex for ex in [find_in(child, fn) for child in expr.children] if ex is not None]
    return exprs[0] if exprs else None


def top_level_instance_of(expr: ITeg, fn):
    return find_in(expr, fn)


def transfer_bounds_deriv_mode(expr: BiMap, source_lower: Dict[TegVar, ITeg], source_upper: Dict[TegVar, ITeg]):
    """Implements a derivative-based pessimistic bounds computation for
    continuous monotonic maps. """
    lb_lets = {}
    ub_lets = {}

    for tegvar in source_lower:
        deriv_expr = fwd_deriv(expr, {tegvar: Const(1)})
        lb_lets[tegvar] = (IfElse(deriv_expr > 0, source_upper[tegvar], source_lower[tegvar]))
        ub_lets[tegvar] = (IfElse(deriv_expr > 0, source_lower[tegvar], source_upper[tegvar]))

    return LetIn(lb_lets.keys(), lb_lets.values(), expr), LetIn(ub_lets.keys(), ub_lets.values(), expr)


def transfer_bounds_general(expr: BiMap, source_lower: Dict[TegVar, ITeg], source_upper: Dict[TegVar, ITeg]):
    """Implements a derivative-based pessimistic bounds computation for
    continuous monotonic maps. """
    lb_lets = {}
    ub_lets = {}

    for tegvar in source_lower:
        deriv_expr = fwd_deriv(expr, {tegvar: Const(1)})
        lb_lets[tegvar] = (IfElse(deriv_expr > 0, source_upper[tegvar], source_lower[tegvar]))
        ub_lets[tegvar] = (IfElse(deriv_expr > 0, source_lower[tegvar], source_upper[tegvar]))

    return LetIn(lb_lets.keys(), lb_lets.values(), expr), LetIn(ub_lets.keys(), ub_lets.values(), expr)


def resolve_placeholders(expr: ITeg,
                         map: Dict[str, ITeg]):
    """Substitute placeholders for their expressions. """
    for key, p_expr in map.items():
        expr = substitute(expr, Placeholder(signature=key), p_expr)

    return expr


def extract_vars(expr: ITeg):
    if isinstance(expr, (Const, Placeholder)):
        return set()

    if isinstance(expr, Var):
        return {expr}

    if hasattr(expr, 'children'):
        return reduce(operator.or_, [extract_vars(child) for child in expr.children], set())
    else:
        return set()
