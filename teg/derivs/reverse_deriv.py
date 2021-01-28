from typing import Set, List, Tuple, Iterable, Optional, Dict, Any
from collections import defaultdict
from functools import reduce
import operator

from teg import (
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
    SmoothFunc,
    true,
    false
)
from teg.lang.extended import Delta, BiMap
from teg.lang.extended_utils import extract_vars

from teg.passes.substitute import substitute

from .edge.common import primitive_booleans_in, extend_dependencies


def cache(f):
    cache = {}

    def wrapper_f(expr, out_deriv_vals, not_ctx, deps, args):

        k = (expr, out_deriv_vals)
        if k not in cache:
            cache[k] = list(f(expr, out_deriv_vals, not_ctx, deps, args))
        return cache[k]
    return wrapper_f


def merge(a, b, multiplier):
    parts = a + b
    parts_dict = {}
    for i in parts:
        parts_dict.setdefault(i[0], []).append(i[1])

    all_list = []
    for (uid, exprs) in parts_dict.items():
        all_list.append((uid, multiplier * sum(exprs)))

    yield from all_list


@cache
def reverse_deriv_transform(expr: ITeg,
                            out_deriv_vals: Tuple,
                            not_ctx: Set[Tuple[str, int]],
                            deps: Dict[TegVar, Set[Var]],
                            args: Dict[str, Any]) -> Iterable[Tuple[Tuple[str, int], ITeg]]:

    if isinstance(expr, TegVar):
        if (((expr.name, expr.uid) not in not_ctx) or
            {(v.name, v.uid) for v in extend_dependencies({expr}, deps)} - not_ctx):
            yield ((f'd{expr.name}', expr.uid), out_deriv_vals)

    elif isinstance(expr, (Const, Delta)):
        pass

    elif isinstance(expr, Var):
        if (expr.name, expr.uid) not in not_ctx:
            yield ((f'd{expr.name}', expr.uid), out_deriv_vals)

    elif isinstance(expr, Add):
        left, right = expr.children
        left_list = list(reverse_deriv_transform(left, Const(1), not_ctx, deps, args))
        right_list = list(reverse_deriv_transform(right, Const(1), not_ctx, deps, args))
        yield from merge(left_list, right_list, out_deriv_vals)

    elif isinstance(expr, Mul):
        left, right = expr.children
        left_list = list(reverse_deriv_transform(left, right, not_ctx, deps, args))
        right_list = list(reverse_deriv_transform(right, left, not_ctx, deps, args))
        yield from merge(left_list, right_list, out_deriv_vals)

    elif isinstance(expr, Invert):
        child = expr.child
        yield from reverse_deriv_transform(child, -out_deriv_vals * expr * expr, not_ctx, deps, args)

    elif isinstance(expr, SmoothFunc):
        child = expr.expr
        yield from reverse_deriv_transform(child, expr.rev_deriv(out_deriv_expr=out_deriv_vals), not_ctx, deps, args)

    elif isinstance(expr, IfElse):
        derivs_if = reverse_deriv_transform(expr.if_body, Const(1), not_ctx, deps, args)
        derivs_else = reverse_deriv_transform(expr.else_body, Const(1), not_ctx, deps, args)
        yield from ((name_uid, out_deriv_vals * IfElse(expr.cond, deriv_if, Const(0)))
                    for name_uid, deriv_if in derivs_if)
        yield from ((name_uid, out_deriv_vals * IfElse(expr.cond, Const(0), deriv_else))
                    for name_uid, deriv_else in derivs_else)

        if not args.get('ignore_deltas', False):
            for boolean in primitive_booleans_in(expr.cond, not_ctx, deps):
                jump = substitute(expr, boolean, true) - substitute(expr, boolean, false)
                delta_expr = boolean.right_expr - boolean.left_expr
                derivs_delta_expr = reverse_deriv_transform(delta_expr, Const(1), not_ctx, deps, args)
                yield from ((name_uid, out_deriv_vals * deriv_delta_expr * jump * Delta(delta_expr))
                            for name_uid, deriv_delta_expr in derivs_delta_expr)

    elif isinstance(expr, Teg):
        not_ctx.discard((expr.dvar.name, expr.dvar.uid))

        if not args.get('ignore_bounds', False):
            lower_derivs = reverse_deriv_transform(expr.lower, out_deriv_vals, not_ctx, deps, args)
            upper_derivs = reverse_deriv_transform(expr.upper, out_deriv_vals, not_ctx, deps, args)
            yield from ((name_uid, upper_deriv * substitute(expr.body, expr.dvar, expr.upper))
                        for name_uid, upper_deriv in upper_derivs)
            yield from ((name_uid, - lower_deriv * substitute(expr.body, expr.dvar, expr.lower))
                        for name_uid, lower_deriv in lower_derivs)

        not_ctx.add((expr.dvar.name, expr.dvar.uid))

        deriv_body_traces = reverse_deriv_transform(expr.body,
                                                    Const(1),
                                                    not_ctx,
                                                    deps,
                                                    args)

        yield from ((name_uid, out_deriv_vals * Teg(expr.lower, expr.upper, deriv_body, expr.dvar))
                    for name_uid, deriv_body in deriv_body_traces)

    elif isinstance(expr, Tup):
        yield [reverse_deriv_transform(child, out_deriv_vals, not_ctx, deps, args)
               for child in expr]

    elif isinstance(expr, LetIn):
        dnew_vars, body_derivs = set(), {}
        for var, e in zip(expr.new_vars, expr.new_exprs):


            if any(Var(name=ctx_name, uid=ctx_uid) in e for ctx_name, ctx_uid in not_ctx):
                assert isinstance(var, TegVar), f'{var} is dependent on TegVar(s):'\
                                                f'({[ctx_var for ctx_var in not_ctx if ctx_var in e]}).'\
                                                f'{var} must also be declared as a TegVar and not a Var'
    
                not_ctx = not_ctx | {(var.name, var.uid)}


            if var not in expr.expr:
                continue

            dname = f'd{var.name}'
            dnew_vars.add((dname, var.uid))
            body_derivs[(dname, var.uid)] = list(reverse_deriv_transform(e, Const(1), not_ctx, deps, args))

        for (name, uid), dname_expr in reverse_deriv_transform(expr.expr, out_deriv_vals, not_ctx, deps, args):
            dvar_with_ctx = LetIn(expr.new_vars, expr.new_exprs, dname_expr)
            if (name, uid) in dnew_vars:
                yield from ((n, d * dvar_with_ctx) for n, d in body_derivs[(name, uid)])
            else:
                yield ((name, uid), dvar_with_ctx)

    elif isinstance(expr, BiMap):
        dnew_vars, body_derivs = set(), {}
        new_deps = {}
        for var, e in zip(expr.targets, expr.target_exprs):


            if any(Var(name=ctx_name, uid=ctx_uid) in e for ctx_name, ctx_uid in not_ctx):
    
                assert isinstance(var, TegVar), f'{var} is dependent on TegVar(s):'\
                                                f'({[ctx_var for ctx_var in not_ctx if ctx_var in e]}).'\
                                                f'{var} must also be declared as a TegVar and not a Var'
    
                not_ctx = not_ctx | {(var.name, var.uid)}

            if var not in expr.expr:
    
                continue
            new_deps[var] = extract_vars(e)

            dname = f'd{var.name}'
            dnew_vars.add((dname, var.uid))
            body_derivs[(dname, var.uid)] = list(reverse_deriv_transform(e, Const(1), not_ctx, deps, args))

        deps = {**deps, **new_deps}
        for (name, uid), dname_expr in reverse_deriv_transform(expr.expr, out_deriv_vals, not_ctx, deps, args):
            dvar_with_ctx = BiMap(dname_expr,
                                  expr.targets, expr.target_exprs,
                                  expr.sources, expr.source_exprs,
                                  inv_jacobian=expr.inv_jacobian,
                                  target_lower_bounds=expr.target_lower_bounds,
                                  target_upper_bounds=expr.target_upper_bounds)
            if (name, uid) in dnew_vars:
                yield from ((n, d * dvar_with_ctx) for n, d in body_derivs[(name, uid)])
            else:
                yield ((name, uid), dvar_with_ctx)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')


def reverse_deriv(expr: ITeg, out_deriv_vals: Tup = None,
                  output_list: Optional[List[Var]] = None, args: Dict[str, Any] = None) -> ITeg:
    """Computes the derivative of a given expression.

    Args:
        expr: The expression to compute the total derivative of.
        out_deriv_vals: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The reverse derivative expression.
    """

    if out_deriv_vals is None:
        out_deriv_vals = Tup(Const(1))

    if args is None:
        args = {}

    def derivs_for_single_outval(expr: ITeg,
                                 single_outval: Const,
                                 i: Optional[int] = None,
                                 output_list: Optional[List[Var]] = None,
                                 args: Dict[str, Any] = None) -> Tuple[List[Var], ITeg]:

        partial_deriv_map = defaultdict(lambda: Const(0))

        for name_uid, e in reverse_deriv_transform(expr, single_outval, set(), {}, args):
            partial_deriv_map[name_uid] += e

        uids = [var_uid for var_name, var_uid in partial_deriv_map.keys()]
        new_vars = [Var(var_name) for var_name, var_uid in partial_deriv_map.keys()]
        new_vals = [*partial_deriv_map.values()]

        if output_list is not None:

            var_map = {uid: (var, val) for uid, var, val in zip(uids, new_vars, new_vals)}
            new_vars, new_vals = zip(*[var_map.get(var.uid, (Var(f'd{var.name}'), Const(0)))
                                       for var in output_list])
        else:

            sorted_list = list(zip(uids, new_vars, new_vals))
            sorted_list.sort(key=lambda a: a[0])
            _, new_vars, new_vals = list(zip(*sorted_list))


        assert len(new_vals) > 0, 'There must be variables to compute derivatives. '
        return new_vars, (Tup(*new_vals) if len(new_vars) > 1 else new_vals[0])

    if len(out_deriv_vals) == 1:
        single_outval = out_deriv_vals.children[0]
        derivs = derivs_for_single_outval(expr, single_outval, 0, output_list=output_list, args=args)
    else:
        assert len(out_deriv_vals) == len(expr), \
            f'Expected out_deriv to have "{len(expr)}" values, but got "{len(out_deriv_vals)}" values.'

        derivs = (derivs_for_single_outval(e, single_outval, i, output_list=output_list, args=args)
                  for i, (e, single_outval) in enumerate(zip(expr, out_deriv_vals)))
        derivs = Tup(*derivs)
    return derivs
