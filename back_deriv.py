from typing import Dict, Set, Tuple, Callable, Iterable
from collections import defaultdict

from integrable_program import (
    Teg,
    TegConstant,
    TegVariable,
    TegAdd,
    TegMul,
    TegConditional,
    TegIntegral,
)


def back_deriv_transform(expr: Teg,
                         dwdout: TegVariable,
                         not_ctx: Set[str]) -> Iterable[Tuple[str, Teg]]:
    # print(expr.name, expr, dwdout, ctx.items())

    if isinstance(expr, TegConstant):
        yield from []

    elif isinstance(expr, TegVariable):
        if expr.name not in not_ctx:
            yield (f'd{expr.name}', dwdout)
        yield from []

    elif isinstance(expr, TegAdd):
        left, right = expr.children
        yield from back_deriv_transform(left, dwdout, not_ctx)
        yield from back_deriv_transform(right, dwdout, not_ctx)

    elif isinstance(expr, TegMul):
        left, right = expr.children
        yield from back_deriv_transform(left, dwdout * right, not_ctx)
        yield from back_deriv_transform(right, dwdout * left, not_ctx)

    elif isinstance(expr, TegConditional):
        derivs_if = back_deriv_transform(expr.if_body, TegConstant(1), not_ctx)
        derivs_else = back_deriv_transform(expr.else_body, TegConstant(1), not_ctx)
        yield from ((name, TegConditional(expr.var, expr.const, deriv_if, TegConstant(0)))
                    for name, deriv_if in derivs_if)
        yield from ((name, dwdout * TegConditional(expr.var, expr.const, TegConstant(0), deriv_else))
                    for name, deriv_else in derivs_else)

    elif isinstance(expr, TegIntegral):
        not_ctx.add(expr.dvar.name)
        deriv_body_traces = back_deriv_transform(expr.body, TegConstant(1), not_ctx)
        yield from ((name, dwdout * TegIntegral(expr.lower, expr.upper, deriv_body, expr.dvar))
                    for name, deriv_body in deriv_body_traces)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')


def back_deriv(expr: Teg, out_deriv_val: float = 1):
    """Computes the derivative of a given expression.

    Args:
        expr: The expression to compute the total derivative of.
        out_deriv_val: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The reverse derivative expression.
    """
    partial_deriv_map = defaultdict(lambda: TegConstant(0))

    # After deriv_transform, expr will have unbound infinitesimals
    for name, e in back_deriv_transform(expr, TegVariable("dnewdout", out_deriv_val), set()):
        partial_deriv_map[name] += e

    return partial_deriv_map
