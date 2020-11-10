from typing import Set, Tuple, Iterable, Optional
from collections import defaultdict

from integrable_program import (
    ITeg,
    Const,
    Var,
    TegVar,
    Add,
    Mul,
    Invert,
    IfElse,
    Teg,
    Tup,
    LetIn,
    SmoothFunc
)
from fwd_deriv import delta_contribution, rotated_delta_contribution
from substitute import substitute
from remap import remap, is_remappable
import time

def reverse_deriv_transform(expr: ITeg,
                            out_deriv_vals: Tuple,
                            not_ctx: Set[Tuple[str, int]],
                            teg_list: Set[Tuple[TegVar, ITeg, ITeg]]) -> Iterable[Tuple[Tuple[str, int], ITeg]]:

    if isinstance(expr, (Const, TegVar)):
        pass

    elif isinstance(expr, Var):
        if (expr.name, expr.uid) not in not_ctx:
            yield ((f'd{expr.name}', expr.uid), out_deriv_vals)

    elif isinstance(expr, Add):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals, not_ctx, teg_list)
        yield from reverse_deriv_transform(right, out_deriv_vals, not_ctx, teg_list)

    elif isinstance(expr, Mul):
        left, right = expr.children
        yield from reverse_deriv_transform(left, out_deriv_vals * right, not_ctx, teg_list)
        yield from reverse_deriv_transform(right, out_deriv_vals * left, not_ctx, teg_list)

    elif isinstance(expr, Invert):
        child = expr.child
        yield from reverse_deriv_transform(child, -out_deriv_vals * expr * expr, not_ctx, teg_list)

    elif isinstance(expr, SmoothFunc):
        child = expr.expr
        yield from reverse_deriv_transform(child, expr.rev_deriv(out_deriv_expr = out_deriv_vals), not_ctx, teg_list)

    elif isinstance(expr, IfElse):
        derivs_if = reverse_deriv_transform(expr.if_body, Const(1), not_ctx, teg_list)
        derivs_else = reverse_deriv_transform(expr.else_body, Const(1), not_ctx, teg_list)
        yield from ((name_uid, out_deriv_vals * IfElse(expr.cond, deriv_if, Const(0)))
                    for name_uid, deriv_if in derivs_if)
        yield from ((name_uid, out_deriv_vals * IfElse(expr.cond, Const(0), deriv_else))
                    for name_uid, deriv_else in derivs_else)

    elif isinstance(expr, Teg):
        not_ctx.discard((expr.dvar.name, expr.dvar.uid))
        #moving_var_data = delta_contribution(expr, not_ctx)
        delta_set = rotated_delta_contribution(expr, not_ctx, teg_list | {(expr.dvar, expr.lower, expr.upper)})

        delta_deriv_list = ((name_uid, remapping(out_deriv_vals * delta_expr * deriv_expr))
                    for delta_expr, distance_to_delta, remapping in delta_set
                    for name_uid, deriv_expr in reverse_deriv_transform(distance_to_delta, Const(1), not_ctx, teg_list))

        yield from delta_deriv_list

        # Apply Leibniz rule directly for moving boundaries
        lower_derivs = reverse_deriv_transform(expr.lower, out_deriv_vals, not_ctx, teg_list | {(expr.dvar, expr.lower, expr.upper)})
        upper_derivs = reverse_deriv_transform(expr.upper, out_deriv_vals, not_ctx, teg_list | {(expr.dvar, expr.lower, expr.upper)})
        yield from ((name_uid, upper_deriv * substitute(expr.body, expr.dvar, expr.upper))
                    for name_uid, upper_deriv in upper_derivs)
        yield from ((name_uid, - lower_deriv * substitute(expr.body, expr.dvar, expr.lower))
                    for name_uid, lower_deriv in lower_derivs)

        not_ctx.add((expr.dvar.name, expr.dvar.uid))
        deriv_body_traces = reverse_deriv_transform(expr.body, Const(1), 
                                            not_ctx, teg_list | {(expr.dvar, expr.lower, expr.upper)})
        yield from ((name_uid, out_deriv_vals * Teg(expr.lower, expr.upper, deriv_body, expr.dvar))
                    for name_uid, deriv_body in deriv_body_traces)

    elif isinstance(expr, Tup):
        yield [reverse_deriv_transform(child, out_deriv_vals, not_ctx, teg_list)
               for child in expr]

    elif isinstance(expr, LetIn):
        # Include derivatives of each expression to the let body
        dnew_vars, body_derivs = set(), {}
        for var, e in zip(expr.new_vars, expr.new_exprs):
            dname = f'd{var.name}'
            dnew_vars.add(dname)
            body_derivs[dname] = reverse_deriv_transform(e, Const(1), not_ctx, teg_list)

        # Thread through derivatives of each subexpression
        for (name, uid), dname_expr in reverse_deriv_transform(expr.expr, out_deriv_vals, not_ctx, teg_list):
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
    def remap_all(expr: ITeg):
        while is_remappable(expr):
            expr = remap(expr)
        return expr

    def derivs_for_single_outval(expr: ITeg,
                                 single_outval: Const,
                                 i: Optional[int] = None) -> ITeg:
        partial_deriv_map = defaultdict(lambda: Const(0))

        # After deriv_transform, expr will have unbound infinitesimals
        for name_uid, e in reverse_deriv_transform(expr, single_outval, set(), set()):
            partial_deriv_map[name_uid] += e

        # Introduce fresh variables for each partial derivative
        uids =     [var_uid for var_name, var_uid in partial_deriv_map.keys()]
        new_vars = [Var(var_name) for var_name, var_uid in partial_deriv_map.keys()]
        new_vals = [*partial_deriv_map.values()]
        new_vals = [remap_all(e) for e in new_vals]

        sorted_list = list(zip(uids, new_vars, new_vals))
        sorted_list.sort(key = lambda a:a[0])
        _, new_vars, new_vals = list(zip(*sorted_list))

        print('Reverse-deriv output order: ', new_vars)

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
