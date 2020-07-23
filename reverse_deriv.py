from typing import Set, Tuple, Iterable, Optional
from collections import defaultdict

from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Cond,
    Teg,
    Tup,
    LetIn,
)
from fwd_deriv import delta_contribution
from substitute import substitute


def reverse_deriv_transform(expr: ITeg,
                            out_deriv_vals: Tuple,
                            not_ctx: Set[Tuple[str, int]]) -> Iterable[Tuple[Tuple[str, int], ITeg]]:

    if isinstance(expr, Const):
        yield from []

    elif isinstance(expr, Var):
        if (expr.name, expr.uid) not in not_ctx:
            yield ((f'd{expr.name}', expr.uid), out_deriv_vals)
        yield from []

    elif isinstance(expr, Add):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals, not_ctx)
        yield from reverse_deriv_transform(right, out_deriv_vals, not_ctx)

    elif isinstance(expr, Mul):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals * right, not_ctx)
        yield from reverse_deriv_transform(right, out_deriv_vals * left, not_ctx)

    elif isinstance(expr, Cond):
        derivs_if = reverse_deriv_transform(expr.if_body, Const(1), not_ctx)
        derivs_else = reverse_deriv_transform(expr.else_body, Const(1), not_ctx)
        yield from ((name_uid, out_deriv_vals * Cond(expr.lt_expr, deriv_if, Const(0)))
                    for name_uid, deriv_if in derivs_if)
        yield from ((name_uid, out_deriv_vals * Cond(expr.lt_expr, Const(0), deriv_else))
                    for name_uid, deriv_else in derivs_else)

    elif isinstance(expr, Teg):
        not_ctx.discard((expr.dvar.name, expr.dvar.uid))
        moving_var_data = delta_contribution(expr, not_ctx)
        yield from (((dname, uid), out_deriv_vals * delta_val)
                    for (name, uid), (dname, delta_val) in moving_var_data.items())

        # Apply Leibniz rule directly for moving boundaries
        lower_derivs = reverse_deriv_transform(expr.lower, out_deriv_vals, not_ctx)
        upper_derivs = reverse_deriv_transform(expr.upper, out_deriv_vals, not_ctx)
        yield from ((name_uid, upper_deriv * substitute(expr.body, expr.dvar, upper_deriv))
                    for name_uid, upper_deriv in upper_derivs)
        yield from ((name_uid, - lower_deriv * substitute(expr.body, expr.dvar, lower_deriv))
                    for name_uid, lower_deriv in lower_derivs)

        not_ctx.add((expr.dvar.name, expr.dvar.uid))
        deriv_body_traces = reverse_deriv_transform(expr.body, Const(1), not_ctx)
        yield from ((name_uid, out_deriv_vals * Teg(expr.lower, expr.upper, deriv_body, expr.dvar))
                    for name_uid, deriv_body in deriv_body_traces)

    elif isinstance(expr, Tup):
        yield [reverse_deriv_transform(child, out_deriv_vals, not_ctx)
               for child in expr]

    elif isinstance(expr, LetIn):
        # Include derivatives of each expression to the let body
        dnew_vars, body_derivs = set(), {}
        for var, e in zip(expr.new_vars, expr.new_exprs):
            dname = f'd{var.name}'
            dnew_vars.add(dname)
            body_derivs[dname] = reverse_deriv_transform(e, Const(1), not_ctx)

        # Thread through derivatives of each subexpression
        for (name, uid), dname_expr in reverse_deriv_transform(expr.expr, out_deriv_vals, not_ctx):
            dvar_with_ctx = LetIn(expr.new_vars, expr.new_exprs, dname_expr)
            if name in dnew_vars:
                yield from ((n, d * dvar_with_ctx) for n, d in body_derivs[name])
            else:
                yield ((name, uid), dvar_with_ctx)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')


def reverse_deriv(expr: ITeg, out_deriv_vals: Tup) -> ITeg:
    """Computes the derivative of a given expression.

    Args:
        expr: The expression to compute the total derivative of.
        out_deriv_vals: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The reverse derivative expression.
    """
    def derivs_for_single_outval(expr: ITeg,
                                 single_outval: Const,
                                 i: Optional[int] = None) -> ITeg:
        partial_deriv_map = defaultdict(lambda: Const(0))

        # After deriv_transform, expr will have unbound infinitesimals
        for name_uid, e in reverse_deriv_transform(expr, single_outval, set()):
            partial_deriv_map[name_uid] += e

        # Introduce fresh variables for each partial derivative
        new_vars = [Var(var_name) for var_name, var_uid in partial_deriv_map.keys()]
        new_vals = [*partial_deriv_map.values()]

        assert len(new_vals) > 0, 'There must be variables to compute derivatives. '
        out_expr = Tup(*new_vars) if len(new_vars) > 1 else new_vars[0]
        derivs = LetIn(Tup(*new_vars), Tup(*new_vals), out_expr)
        return derivs

    if len(out_deriv_vals) == 1:
        single_outval = out_deriv_vals.children[0]
        derivs = derivs_for_single_outval(expr, single_outval, 0)
    else:
        assert len(out_deriv_vals) == len(expr), \
            f'Expected out_deriv to have "{len(expr)}" values, but got "{len(out_deriv_vals)}" values.'

        derivs = (derivs_for_single_outval(e, single_outval, i)
                  for i, (e, single_outval) in enumerate(zip(expr, out_deriv_vals)))
        derivs = Tup(*derivs)
    return derivs
