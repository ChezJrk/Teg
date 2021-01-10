from typing import Dict, Set, List, Tuple
from functools import reduce

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
    # TegRemap
)
from teg.passes.substitute import substitute
# from teg.passes.remap import remap, is_remappable

# from .edge.rotated import rotated_delta_contribution
from .edge.common import primitive_booleans_in


def boundary_contribution(expr: ITeg,
                          ctx: Dict[Tuple[str, int], ITeg],
                          not_ctx: Set[Tuple[str, int]]
                          ) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """ Apply Leibniz rule directly for moving boundaries. """
    lower_deriv, ctx1, not_ctx1, _ = fwd_deriv_transform(expr.lower, ctx, not_ctx, set())
    upper_deriv, ctx2, not_ctx2, _ = fwd_deriv_transform(expr.upper, ctx, not_ctx, set())
    body_at_upper = substitute(expr.body, expr.dvar, expr.upper)
    body_at_lower = substitute(expr.body, expr.dvar, expr.lower)

    boundary_val = upper_deriv * body_at_upper - lower_deriv * body_at_lower
    return boundary_val, {**ctx1, **ctx2}, not_ctx1 | not_ctx2


def fwd_deriv_transform(expr: ITeg,
                        ctx: Dict[Tuple[str, int], ITeg],
                        not_ctx: Set[Tuple[str, int]],
                        teg_list: Set[Tuple[TegVar, ITeg, ITeg]]
                        ) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """Compute the source-to-source foward derivative of the given expression."""
    if isinstance(expr, TegVar):
        if (expr.name, expr.uid) not in not_ctx:
            if (expr.name, expr.uid) not in ctx:

                # Introduce derivative and leave the value unbound
                var = Var(f'd{expr.name}')
                ctx[(expr.name, expr.uid)] = var
                # print('TEGVAR-GEN')
                # print(f'{expr.name}: {ctx}')
            else:
                # Use the old derivative
                var = ctx[(expr.name, expr.uid)]
            expr = var
        else:
            expr = Const(0)

    elif isinstance(expr, (Const, Placeholder)):
        expr = Const(0)

    # elif isinstance(expr, (TegRemap,)):
    #    assert False, "Cannot take derivative of transient elements."

    elif isinstance(expr, BiMap):
        # No derivative requirement.
        # BiMaps are only for TegVars and they shouldn't have derivatives.
        pass

    elif isinstance(expr, Delta):
        assert False, "Cannot take the derivative of a Delta. Reduce all Deltas first"

    elif isinstance(expr, Var):
        # NOTE: No expressions with the name "d{expr.name}" are allowed
        if (expr.name, expr.uid) not in not_ctx:
            if (expr.name, expr.uid) not in ctx:
                # Introduce derivative and leave the value unbound
                var = Var(f'd{expr.name}')
                ctx[(expr.name, expr.uid)] = var
                # print('TEGVAR-GEN')
                # print(f'{expr.name}, {ctx}')
            else:
                # Use the old derivative
                var = ctx[(expr.name, expr.uid)]
            expr = var
        else:
            expr = Const(0)

    elif isinstance(expr, SmoothFunc):
        in_deriv_expr, ctx, not_ctx, teg_list = fwd_deriv_transform(expr.expr, ctx, not_ctx, teg_list)
        deriv_expr = expr.fwd_deriv(in_deriv_expr=in_deriv_expr)
        expr = deriv_expr

    elif isinstance(expr, Add):
        def union(out1, out2):
            e1, ctx1, not_ctx1, teg_list1 = out1
            e2, ctx2, not_ctx2, teg_list2 = out2
            return expr.operation(e1, e2), {**ctx1, **ctx2}, not_ctx1 | not_ctx2, teg_list1 | teg_list2

        e_all = Const(0)
        for child in expr.children:
            e, ctx, not_ctx, teg_list = fwd_deriv_transform(
                                        child, ctx,
                                        not_ctx, teg_list
                                    )
            e_all = e_all + e

        expr = e_all

    elif isinstance(expr, Mul):
        # NOTE: Consider n-ary multiplication.
        assert len(expr.children) == 2, 'fwd_deriv does not currently handle non-binary multiplication'
        expr1, expr2 = [child for child in expr.children]

        (fwd_deriv_expr1, ctx1, not_ctx1, _) = fwd_deriv_transform(
                                                    expr1, ctx,
                                                    not_ctx, teg_list
                                               )

        (fwd_deriv_expr2, ctx2, not_ctx2, _) = fwd_deriv_transform(
                                                    expr2, ctx,
                                                    not_ctx, teg_list
                                               )

        expr = expr1 * fwd_deriv_expr2 + expr2 * fwd_deriv_expr1
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2

    elif isinstance(expr, Invert):
        deriv_expr, ctx, not_ctx, teg_list = fwd_deriv_transform(expr.child, ctx, not_ctx, teg_list)
        expr = -expr * expr * deriv_expr

    elif isinstance(expr, IfElse):
        if_body, ctx, not_ctx1, _ = fwd_deriv_transform(expr.if_body, ctx, not_ctx, teg_list)
        else_body, ctx, not_ctx2, _ = fwd_deriv_transform(expr.else_body, ctx, not_ctx, teg_list)
        # ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2

        deltas = Const(0)
        for boolean in primitive_booleans_in(expr.cond, not_ctx):
            jump = substitute(expr, boolean, true) - substitute(expr, boolean, false)
            delta_expr = boolean.right_expr - boolean.left_expr

            delta_deriv, ctx, _ignore_not_ctx, _ = fwd_deriv_transform(delta_expr, ctx, not_ctx, teg_list)
            deltas = deltas + delta_deriv * jump * Delta(delta_expr)

        expr = IfElse(expr.cond, if_body, else_body) + deltas

    elif isinstance(expr, Teg):
        assert expr.dvar not in ctx, f'Names of infinitesimal "{expr.dvar}" are distinct from context "{ctx}"'
        not_ctx.discard(expr.dvar.name)  # TODO: Why is this here?

        # Include derivative contribution from moving boundaries of integration
        boundary_val, new_ctx, new_not_ctx = boundary_contribution(expr, ctx, not_ctx)
        not_ctx.add((expr.dvar.name, expr.dvar.uid))

        """
        moving_var_data = delta_contribution(expr, not_ctx)
        for (moving_var_delta, expr_for_dvar) in moving_var_data:
            print(f"Delta: {moving_var_delta}, \n Point-Deriv: {expr_for_dvar} \n")
            deriv_expr, ctx, not_ctx = fwd_deriv_transform(expr_for_dvar, ctx, not_ctx)
            delta_val += deriv_expr * moving_var_delta
        """

        """
        delta_val = Const(0)

        delta_set = rotated_delta_contribution(expr, not_ctx, teg_list | {(expr.dvar, expr.lower, expr.upper)})
        for delta_expression, distance_to_delta, remapping in delta_set:
            distance_derivative, ctx, not_ctx, _ = fwd_deriv_transform(distance_to_delta, ctx, not_ctx, set())
            delta_val += remapping(delta_expression * distance_derivative)
        """

        body, ctx, not_ctx, _ = fwd_deriv_transform(expr.body, ctx, not_ctx,
                                                    teg_list | {(expr.dvar, expr.lower, expr.upper)})

        ctx.update(new_ctx)
        not_ctx |= new_not_ctx
        expr = Teg(expr.lower, expr.upper, body, expr.dvar) + boundary_val  # + delta_val + boundary_val

    elif isinstance(expr, Tup):
        new_expr_list, new_ctx, new_not_ctx = [], Ctx(), set()
        for child in expr:
            child, ctx, not_ctx, _ = fwd_deriv_transform(child, ctx, not_ctx, teg_list)
            new_expr_list.append(child)
            new_ctx.update(ctx)
            new_not_ctx |= not_ctx
        ctx, not_ctx = new_ctx, new_not_ctx
        expr = Tup(*new_expr_list)

    elif isinstance(expr, LetIn):

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = list(expr.new_vars), list(expr.new_exprs)
        for v, e in zip(expr.new_vars, expr.new_exprs):
            # By not passing in the updated contexts, require independence of exprs in the body of the let expression
            de, ctx, not_ctx, _ = fwd_deriv_transform(e, ctx, not_ctx, teg_list)
            ctx[(v.name, v.uid)] = Var(f'd{v.name}')
            new_vars_with_derivs.append(ctx[(v.name, v.uid)])
            new_exprs_with_derivs.append(de)

        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we
        # remove them from ctx!
        dexpr, ctx, not_ctx, _ = fwd_deriv_transform(expr.expr, ctx, not_ctx, teg_list)
        [ctx.pop((c.name, c.uid), None) for c in expr.new_vars]

        expr = LetIn(Tup(*new_vars_with_derivs), Tup(*new_exprs_with_derivs), dexpr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')

    # print(ctx)
    return expr, ctx, not_ctx, teg_list


def fwd_deriv(expr: ITeg, bindings: List[Tuple[ITeg, int]]) -> ITeg:
    """
    Computes the fwd_derivative of a given expression.

    Args:
        expr: The expression to compute the total fwd_derivative of.
        bindings: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The forward fwd_derivative expression.
    """
    binding_map = {(var.name, var.uid): val for var, val in bindings}

    # print('OLD: ')
    # print(expr)
    # After fwd_deriv_transform, expr will have unbound infinitesimals
    full_expr, ctx, not_ctx, _ = fwd_deriv_transform(expr, {}, set(), set())

    # Resolve all TegRemap expressions by lifting expressions
    # out of the tree.
    expr = full_expr
    # while is_remappable(expr):
    #    expr = remap(expr)

    # print('DERIV:')
    # print(expr)
    assert binding_map.keys() == ctx.keys(), (f'You provided bindings for "{set(binding_map.keys())}" '
                                              f'but bindings were produced for "{set(ctx.keys())}"')

    # Bind the infinitesimals introduced by taking the derivative
    for name_uid, new_var in ctx.items():
        expr.bind_variable(new_var, binding_map[name_uid])

    return expr
