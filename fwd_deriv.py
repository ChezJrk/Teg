from typing import Dict, Set, Tuple, Iterable
from functools import reduce
from copy import deepcopy
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


# TODO: Add in lexical scoping at some point (consider the change for eval, reverse, and forward mode)
# Teg does not support expressions of the form integral_x d_x (f(x)).
# I think it should work once I actually start handling contexts properly.
# Currently everything is evaluated in a single global context making
# integral_x d_x (f(x)) and d_x integral_x (f(x)) indistinguishable.
# Once I have contexts, I can have a separate context for the integral and have it shadow the outer context.


def extract_moving_discontinuities(expr: Teg,
                                   var: TegVariable,
                                   not_ctx: Set[str]) -> Iterable[TegConditional]:
    """Identify all subexpressions producing a moving discontinuity.

    Concretely, this means finding each branching statement including
    the integration variable and a variable defined in an outside context.
    """

    if (isinstance(expr, TegConditional)
            and type(expr.var1) == TegVariable
            and type(expr.var2) == TegVariable
            and (expr.var1.name == var.name or expr.var2.name == var.name)
            and (expr.var1.name not in not_ctx and expr.var2.name not in not_ctx)):
        # if dvar=x and the condition is x<t then returns (expr, t)
        yield (expr, expr.var2 if expr.var1.name == var.name else expr.var1)

    yield from (moving_cond for child in expr.children
                for moving_cond in extract_moving_discontinuities(child, var, not_ctx))


def delta_contribution(expr, not_ctx):
    moving_discontinuities = extract_moving_discontinuities(expr.body, expr.dvar, not_ctx.copy())

    # Avoid overcounting the same discontinuities e.g. [x < 1] + [x < 1] = 2 not 4
    all_discont_exprs, moving_vars = defaultdict(list), defaultdict(list)
    for discont_expr, moving_var in moving_discontinuities:
        all_discont_exprs[moving_var.name].append(discont_expr)
        moving_vars[moving_var.name].append(moving_var)

    # Descends into all subexpressions extracting moving discontinuities
    moving_var_data = {}
    for moving_vars, discont_exprs in zip(moving_vars.values(), all_discont_exprs.values()):
        # Store values before
        allow_eqs_before = [discont_expr.allow_eq for discont_expr in discont_exprs]

        # Evaluate the discontinuity at x = t+
        for discont_expr in discont_exprs:
            discont_expr.allow_eq = False
        expr_body_right = deepcopy(expr.body)
        for moving_var in moving_vars:
            expr_body_right.bind_variable(expr.dvar.name, moving_var.value)

        # Evaluate the discontinuity at x = t-
        for discont_expr in discont_exprs:
            discont_expr.allow_eq = True
        expr_body_left = deepcopy(expr.body)
        for moving_var in moving_vars:
            expr_body_left.bind_variable(expr.dvar.name, moving_var.value)

        # Set the value to what it was before
        for discont_expr, allow_eq_before in zip(discont_exprs, allow_eqs_before):
            discont_expr.allow_eq = allow_eq_before

        # if lower < dvar < upper, include the contribution from the discontinuity
        moving_var_delta = TegConditional(moving_var,
                                          expr.upper,
                                          TegConditional(expr.lower,
                                                         moving_var,
                                                         expr_body_right - expr_body_left,
                                                         TegConstant(0)),
                                          TegConstant(0))
        moving_var_data[moving_var.name] = (f'd{moving_var.name}', moving_var_delta)

    return moving_var_data


def fwd_deriv_transform(expr: Teg, ctx: Dict[str, str], not_ctx: Set[str]) -> Tuple[Teg, Dict[str, str], Set[str]]:

    if isinstance(expr, TegConstant):
        expr = TegConstant(0, name=expr.name)

    elif isinstance(expr, TegVariable):
        # NOTE: No expressions with the name "d{expr.name}" are allowed
        if expr.name not in not_ctx:
            # Introduce derivative and leave the value unbound
            new_name = f'd{expr.name}'
            ctx[expr.name] = new_name
            expr = TegVariable(name=new_name)
        else:
            expr = TegConstant(0)

    elif isinstance(expr, TegAdd):
        def union(out1, out2):
            e1, ctx1, not_ctx1 = out1
            e2, ctx2, not_ctx2 = out2
            return expr.operation(e1, e2), {**ctx1, **ctx2}, not_ctx1 | not_ctx2
        expr, ctx, not_ctx = reduce(union, (fwd_deriv_transform(child, ctx, not_ctx) for child in expr.children))

    elif isinstance(expr, TegMul):
        # NOTE: Consider n-ary multiplication.
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
        expr = TegConditional(expr.var1, expr.var2, if_body, else_body)

    elif isinstance(expr, TegIntegral):
        assert expr.dvar not in ctx, f'Names of infinitesimal "{expr.dvar}" are distinct from context "{ctx}"'
        not_ctx.discard(expr.dvar.name)
        moving_var_data = delta_contribution(expr, not_ctx)
        delta_val = TegConstant(0)
        for name, (new_name, val) in moving_var_data.items():
            ctx[name] = new_name
            delta_val += TegVariable(new_name) * val
        not_ctx.add(expr.dvar.name)
        body, ctx, not_ctx = fwd_deriv_transform(expr.body, ctx, not_ctx)
        expr = TegIntegral(expr.lower, expr.upper, body, expr.dvar) + delta_val

    elif isinstance(expr, TegTuple):
        new_expr_list, new_ctx, new_not_ctx = [], TegContext(), set()
        for child in expr:
            child, ctx, not_ctx = fwd_deriv_transform(child, ctx, not_ctx)
            new_expr_list.append(child)
            new_ctx.update(ctx)
            new_not_ctx |= not_ctx
        ctx, not_ctx = new_ctx, new_not_ctx
        expr = TegTuple(*new_expr_list)

    elif isinstance(expr, TegLetIn):

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = list(expr.new_vars), list(expr.new_exprs)
        for v, e in zip(expr.new_vars, expr.new_exprs):
            # By not passing in the updated contexts, require independence of exprs in the body of the let expression
            de, ctx, not_ctx = fwd_deriv_transform(e, ctx, not_ctx)
            new_vars_with_derivs.append(TegVariable(f'd{v.name}'))
            new_exprs_with_derivs.append(de)

        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we
        # remove them from ctx!
        dexpr, ctx, not_ctx = fwd_deriv_transform(expr.expr, ctx, not_ctx)
        [ctx.pop(c.name, None) for c in expr.new_vars]

        expr = TegLetIn(TegTuple(*new_vars_with_derivs), TegTuple(*new_exprs_with_derivs), dexpr)

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

    assert bindings.keys() == ctx.keys(), f'You must provide bindings for each of the variables "{set(ctx.keys())}"'

    # Bind all free variables to create a valid expression
    for k, new_name in ctx.items():
        expr.bind_variable(new_name, bindings[k])
    return expr
