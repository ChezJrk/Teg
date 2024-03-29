from typing import Dict, Set, List, Tuple

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Invert,
    IfElse,
    Teg,
    Tup,
    LetIn,
    TegVar,
    SmoothFunc,
    Ctx,
    true,
    false
)
from teg.lang.extended import (
    BiMap,
    Delta
)
from teg.lang.markers import (
    Placeholder,
)

from teg.passes.substitute import substitute
from teg.lang.extended_utils import extract_vars
from .edge.common import primitive_booleans_in, extend_dependencies


def boundary_contribution(expr: ITeg,
                          ctx: Dict[Tuple[str, int], ITeg],
                          not_ctx: Set[Tuple[str, int]],
                          deps: Dict[TegVar, Set[Var]]
                          ) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """ Apply Leibniz rule directly for moving boundaries. """
    lower_deriv, ctx1, not_ctx1, _ = fwd_deriv_transform(expr.lower, ctx, not_ctx, deps)
    upper_deriv, ctx2, not_ctx2, _ = fwd_deriv_transform(expr.upper, ctx, not_ctx, deps)
    body_at_upper = substitute(expr.body, expr.dvar, expr.upper)
    body_at_lower = substitute(expr.body, expr.dvar, expr.lower)

    boundary_val = upper_deriv * body_at_upper - lower_deriv * body_at_lower
    return boundary_val, {**ctx1, **ctx2}, not_ctx1 | not_ctx2


def fwd_deriv_transform(expr: ITeg,
                        ctx: Dict[Tuple[str, int], ITeg],
                        not_ctx: Set[Tuple[str, int]],
                        deps: Dict[TegVar, Set[Var]]
                        ) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """Compute the source-to-source foward derivative of the given expression."""
    if isinstance(expr, TegVar):
        if (((expr.name, expr.uid) not in not_ctx
                or {(v.name, v.uid) for v in extend_dependencies({expr}, deps)} - not_ctx)
                and (expr.name, expr.uid) in ctx):
            expr = ctx[(expr.name, expr.uid)]
        else:
            expr = Const(0)

    elif isinstance(expr, (Const, Placeholder, Delta)):
        expr = Const(0)

    elif isinstance(expr, Var):
        if (expr.name, expr.uid) not in not_ctx and (expr.name, expr.uid) in ctx:
            expr = ctx[(expr.name, expr.uid)]
        else:
            expr = Const(0)

    elif isinstance(expr, SmoothFunc):
        in_deriv_expr, ctx, not_ctx, deps = fwd_deriv_transform(expr.expr, ctx, not_ctx, deps)
        deriv_expr = expr.fwd_deriv(in_deriv_expr=in_deriv_expr)
        expr = deriv_expr

    elif isinstance(expr, Add):
        sum_of_derivs = Const(0)
        for child in expr.children:
            deriv_child, ctx, not_ctx, deps = fwd_deriv_transform(child, ctx, not_ctx, deps)
            sum_of_derivs += deriv_child

        expr = sum_of_derivs

    elif isinstance(expr, Mul):
        # NOTE: Consider n-ary multiplication.
        assert len(expr.children) == 2, 'fwd_deriv only supports binary multiplication not n-ary.'
        expr1, expr2 = [child for child in expr.children]

        (deriv_expr1, ctx1, not_ctx1, _) = fwd_deriv_transform(expr1, ctx, not_ctx, deps)
        (deriv_expr2, ctx2, not_ctx2, _) = fwd_deriv_transform(expr2, ctx, not_ctx, deps)

        expr = expr1 * deriv_expr2 + expr2 * deriv_expr1
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2

    elif isinstance(expr, Invert):
        deriv_expr, ctx, not_ctx, deps = fwd_deriv_transform(expr.child, ctx, not_ctx, deps)
        expr = -expr * expr * deriv_expr

    elif isinstance(expr, IfElse):
        if_body, ctx, not_ctx1, _ = fwd_deriv_transform(expr.if_body, ctx, not_ctx, deps)
        else_body, ctx, not_ctx2, _ = fwd_deriv_transform(expr.else_body, ctx, not_ctx, deps)
        not_ctx = not_ctx1 | not_ctx2

        deltas = Const(0)
        for boolean in primitive_booleans_in(expr.cond, not_ctx, deps):
            jump = substitute(expr, boolean, true) - substitute(expr, boolean, false)
            delta_expr = boolean.right_expr - boolean.left_expr

            delta_deriv, ctx, _ignore_not_ctx, _ = fwd_deriv_transform(delta_expr, ctx, not_ctx, deps)
            deltas = deltas + delta_deriv * jump * Delta(delta_expr)

        expr = IfElse(expr.cond, if_body, else_body) + deltas

    elif isinstance(expr, Teg):
        assert expr.dvar not in ctx, f'Names of infinitesimal "{expr.dvar}" are distinct from context "{ctx}"'
        #  In int_x f(x), the variable x is in scope for the integrand f(x)
        not_ctx.discard(expr.dvar.name)

        # Include derivative contribution from moving boundaries of integration
        boundary_val, new_ctx, new_not_ctx = boundary_contribution(expr, ctx, not_ctx, deps)
        not_ctx.add((expr.dvar.name, expr.dvar.uid))

        body, ctx, not_ctx, _ = fwd_deriv_transform(expr.body, ctx, not_ctx, deps)

        ctx.update(new_ctx)
        not_ctx |= new_not_ctx
        expr = Teg(expr.lower, expr.upper, body, expr.dvar) + boundary_val

    elif isinstance(expr, Tup):
        new_expr_list, new_ctx, new_not_ctx = [], Ctx(), set()
        for child in expr:
            child, ctx, not_ctx, _ = fwd_deriv_transform(child, ctx, not_ctx, deps)
            new_expr_list.append(child)
            new_ctx.update(ctx)
            new_not_ctx |= not_ctx
        ctx, not_ctx = new_ctx, new_not_ctx
        expr = Tup(*new_expr_list)

    elif isinstance(expr, LetIn):

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = list(expr.new_vars), list(expr.new_exprs)
        new_deps = {}
        for v, e in zip(expr.new_vars, expr.new_exprs):
            if v in expr.expr:
                # By not passing in the updated contexts,
                # we require that assignments in let expressions are independent
                de, ctx, not_ctx, _ = fwd_deriv_transform(e, ctx, not_ctx, deps)
                ctx[(v.name, v.uid)] = Var(f'd{v.name}')
                new_vars_with_derivs.append(ctx[(v.name, v.uid)])
                new_exprs_with_derivs.append(de)
                new_deps[v] = extract_vars(e)

        deps = {**deps, **new_deps}
        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we
        # remove them from ctx!
        dexpr, ctx, not_ctx, _ = fwd_deriv_transform(expr.expr, ctx, not_ctx, deps)
        [ctx.pop((c.name, c.uid), None) for c in expr.new_vars]

        expr = LetIn(Tup(*new_vars_with_derivs), Tup(*new_exprs_with_derivs), dexpr)

    elif isinstance(expr, BiMap):
        # TODO: is it possible to not repeat this code and make another recursive call instead?

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = [], []
        for v, e in zip(expr.targets, expr.target_exprs):
            if v in expr.expr:
                # By not passing in the updated contexts, require independence of exprs in the body of the let expression
                de, ctx, not_ctx, _ = fwd_deriv_transform(e, ctx, not_ctx, deps)
                ctx[(v.name, v.uid)] = Var(f'd{v.name}')
                new_vars_with_derivs.append(ctx[(v.name, v.uid)])
                new_exprs_with_derivs.append(de)

                not_ctx = not_ctx | {(v.name, v.uid)}

        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we
        # remove them from ctx!
        dexpr, ctx, not_ctx, _ = fwd_deriv_transform(expr.expr, ctx, not_ctx, deps)
        [ctx.pop((c.name, c.uid), None) for c in expr.targets]

        expr = LetIn(Tup(*new_vars_with_derivs), Tup(*new_exprs_with_derivs),
                     BiMap(dexpr,
                           targets=expr.targets,
                           target_exprs=expr.target_exprs,
                           sources=expr.sources,
                           source_exprs=expr.source_exprs,
                           inv_jacobian=expr.inv_jacobian,
                           target_lower_bounds=expr.target_lower_bounds,
                           target_upper_bounds=expr.target_upper_bounds
                           )
                     )

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')

    return expr, ctx, not_ctx, deps


def fwd_deriv(expr: ITeg, bindings: List[Tuple[ITeg, float]], replace_derivs=False) -> ITeg:
    """
    Computes the source-to-source forward of an expression.

    Args:
        expr: An expression that will be differentiated.
        bindings: A mapping from variable names to the values of corresponding infinitesimals.
        replace_derivs: If true, assign derivatives to values specified by bindings.

    Returns:
        ITeg: The forward derivative expression in the extended language.
    """
    binding_map = {(var.name, var.uid): val for var, val in bindings}
    if not replace_derivs:
        # Generate derivative variables
        ctx_map = {(var.name, var.uid): Var(f'd{var.name}') for var, _ in bindings}
    else:
        ctx_map = {(var.name, var.uid): expr for var, expr in bindings}

    # After fwd_deriv_transform, expr will have unbound infinitesimals
    expr, ctx, _, _ = fwd_deriv_transform(expr, ctx_map, set(), {})

    # Bind the infinitesimals introduced by taking the derivative
    if not replace_derivs:
        for name_uid, new_var in ctx.items():
            expr.bind_variable(new_var, binding_map[name_uid])

    return expr
