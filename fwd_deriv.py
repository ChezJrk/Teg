from typing import Dict, Set, Tuple, Callable
from functools import reduce

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


def fwd_deriv_transform(expr: Teg, ctx: Dict[str, str], not_ctx: Set[str]) -> Tuple[Teg, Dict[str, str], Set[str]]:
    # print(expr, ctx, not_ctx)

    if isinstance(expr, TegConstant):
        expr = TegConstant(0, name=expr.name, sign=expr.sign)

    elif isinstance(expr, TegVariable):
        # NOTE: No expressions with the name "d{expr.name}" are allowed
        if expr.name not in not_ctx:
            new_name = f'd{expr.name}'
            ctx[expr.name] = new_name
            expr = TegVariable(name=new_name, value=expr.value, sign=expr.sign)
        else:
            expr = TegConstant(0)

    elif isinstance(expr, TegAdd):
        def union(out1, out2):
            e1, ctx1, not_ctx1 = out1
            e2, ctx2, not_ctx2 = out2
            return expr.operation(e1, e2), {**ctx1, **ctx2}, not_ctx1 | not_ctx2
        expr, ctx, not_ctx = reduce(union, (fwd_deriv_transform(child, ctx, not_ctx) for child in expr.children))

    elif isinstance(expr, TegMul):
        # TODO: For now just implement binop
        # expr, ctx = reduce(lambda e_ctx1, e_ctx2: (expr.operation(e_ctx1[0], e_ctx2[0]), {**e_ctx1[1], **e_ctx2[1]}),
        #                    (fwd_deriv_transform(child, ctx) for child in expr.children))
        expr1, expr2 = [child for child in expr.children]
        deriv = [fwd_deriv_transform(child, ctx, not_ctx) for child in expr.children]
        (fwd_deriv_expr1, ctx1, not_ctx1), (fwd_deriv_expr2, ctx2, not_ctx2) = deriv
        expr = expr1 * fwd_deriv_expr2 + expr2 * fwd_deriv_expr1
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2

    elif isinstance(expr, TegConditional):
        if_body, ctx1, not_ctx1 = fwd_deriv_transform(expr.if_body, ctx, not_ctx)
        else_body, ctx2, not_ctx2 = fwd_deriv_transform(expr.else_body, ctx, not_ctx)
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2
        expr = TegConditional(expr.var, expr.const, if_body, else_body)

    elif isinstance(expr, TegIntegral):
        assert expr.dvar not in ctx, f'Names of infinitesimal "{expr.dvar}" are distinct from context "{ctx}"'
        not_ctx.add(expr.dvar.name)
        body, ctx, not_ctx = fwd_deriv_transform(expr.body, ctx, not_ctx)
        expr = TegIntegral(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, TegTuple):
        new_expr_list, new_ctx, new_not_ctx = [], TegContext(), set()
        for child in expr.children:
            child, ctx, not_ctx = fwd_deriv_transform(child, ctx, not_ctx)
            new_expr_list.append(child)
            new_ctx.update(ctx)
            new_not_ctx |= not_ctx
        ctx, not_ctx = new_ctx, new_not_ctx
        expr = TegTuple(*new_expr_list)

    elif isinstance(expr, TegLetIn):

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = list(expr.new_vars.children), list(expr.new_exprs.children)
        for v, e in zip(expr.new_vars.children, expr.new_exprs.children):
            # By not passing in the updated contexts, require independence of exprs in the body of the let expression
            de, ctx, not_ctx = fwd_deriv_transform(e, ctx, not_ctx)
            new_vars_with_derivs.append(TegVariable(f'd{v.name}'))
            new_exprs_with_derivs.append(de)

        dvar = TegVariable(f'd{expr.var.name}')

        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we 
        # remove them from ctx!
        dexpr, ctx, not_ctx = fwd_deriv_transform(expr.expr, ctx, not_ctx)
        [ctx.pop(c.name, None) for c in expr.new_vars.children]

        expr = TegLetIn(TegTuple(*new_vars_with_derivs), TegTuple(*new_exprs_with_derivs), dvar, dexpr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')

    return expr, ctx, not_ctx


def fwd_deriv(expr: Teg, bindings: TegContext) -> Teg:
    """Computes the fwd_derivative of a given expression.

    Args:
        expr: The expression to compute the total fwd_derivative of.
        bindings: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The forward fwd_derivative expression.
    """

    # After fwd_deriv_transform, expr will have unbound infinitesimals
    expr, ctx, not_ctx = fwd_deriv_transform(expr, {}, set())

    assert bindings.keys() == ctx.keys(), f'You must provide bindings for each of the variables "{ctx.keys()}"'

    # Bind all free variables to create a valid expression
    for k, new_name in ctx.items():
        expr.bind_variable(new_name, bindings[k])
    return expr
