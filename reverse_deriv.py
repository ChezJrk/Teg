from typing import Dict, Set, Tuple, Iterable, Optional
from collections import defaultdict

from integrable_program import (
    Teg,
    TegConstant,
    TegVariable,
    TegAdd,
    TegMul,
    TegConditional,
    TegIntegral,
    TegTuple,
    TegLetIn,
    TegContext,
)


def reverse_deriv_transform(expr: Teg,
                            out_deriv_vals: TegTuple,
                            not_ctx: Set[str]) -> Iterable[Tuple[str, Teg]]:

    if isinstance(expr, TegConstant):
        yield from []

    elif isinstance(expr, TegVariable):
        if expr.name not in not_ctx:
            yield (f'd{expr.name}', out_deriv_vals)
        yield from []

    elif isinstance(expr, TegAdd):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals, not_ctx)
        yield from reverse_deriv_transform(right, out_deriv_vals, not_ctx)

    elif isinstance(expr, TegMul):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals * right, not_ctx)
        yield from reverse_deriv_transform(right, out_deriv_vals * left, not_ctx)

    elif isinstance(expr, TegConditional):
        derivs_if = reverse_deriv_transform(expr.if_body, TegConstant(1), not_ctx)
        derivs_else = reverse_deriv_transform(expr.else_body, TegConstant(1), not_ctx)
        yield from ((name, TegConditional(expr.var, expr.const, deriv_if, TegConstant(0)))
                    for name, deriv_if in derivs_if)
        yield from ((name, out_deriv_vals * TegConditional(expr.var, expr.const, TegConstant(0), deriv_else))
                    for name, deriv_else in derivs_else)

    elif isinstance(expr, TegIntegral):
        not_ctx.add(expr.dvar.name)
        deriv_body_traces = reverse_deriv_transform(expr.body, TegConstant(1), not_ctx)
        yield from ((name, out_deriv_vals * TegIntegral(expr.lower, expr.upper, deriv_body, expr.dvar))
                    for name, deriv_body in deriv_body_traces)

    elif isinstance(expr, TegTuple):
        yield [reverse_deriv_transform(child, out_deriv_vals, not_ctx)
               for child in expr.children]

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')


def reverse_deriv(expr: Teg, out_deriv_vals: TegTuple) -> Teg:
    """Computes the derivative of a given expression.

    Args:
        expr: The expression to compute the total derivative of.
        out_deriv_vals: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The reverse derivative expression.
    """
    def derivs_for_single_outval(expr: Teg,
                                 single_outval: TegConstant,
                                 i: Optional[int] = None) -> Teg:
        partial_deriv_map = defaultdict(lambda: TegConstant(0))

        # After deriv_transform, expr will have unbound infinitesimals
        for name, e in reverse_deriv_transform(expr, single_outval, set()):
            partial_deriv_map[name] += e

        new_vars = [TegVariable(var_name) for var_name in partial_deriv_map.keys()]
        new_vals = [*partial_deriv_map.values()]
        var = TegVariable(f'rev_deriv{i}')
        val = TegTuple(*new_vars) if len(new_vars) > 1 else new_vars[0]
        derivs = TegLetIn(TegTuple(*new_vars), TegTuple(*new_vals), var, val)
        return derivs

    if len(out_deriv_vals.children) == 1:
        single_outval = out_deriv_vals.children[0]
        derivs = derivs_for_single_outval(expr, single_outval, 0)
    else:
        assert len(out_deriv_vals.children) == len(expr.children)

        derivs = (derivs_for_single_outval(e, single_outval, i)
                  for i, (e, single_outval) in enumerate(zip(expr.children, out_deriv_vals.children)))
        derivs = TegTuple(*derivs)

    return derivs
