from teg import (
    ITeg
)

from .base import base_pass


def substitute(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    def inner_fn(e, ctx):
        return e, ctx

    def outer_fn(e, ctx):
        if e == this_expr:
            return that_expr, ctx
        else:
            return e, ctx

    def context_combine(ctxs, ctx):
        return ctx

    n_expr, ctx = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return n_expr


def substitute_instance(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    """
    Substitutes a specific instance of expr (ignores exprs that are symbolically equal to expr but not the same)
    """
    def inner_fn(e, ctx):
        return e, ctx

    def outer_fn(e, ctx):
        if e is this_expr:
            return that_expr, ctx
        else:
            return e, ctx

    def context_combine(ctxs, ctx):
        return ctx

    n_expr, ctx = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return n_expr


"""
def substitute(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:

    if isinstance(expr, Const):
        return expr

    elif expr == this_expr:
        return that_expr

    elif isinstance(expr, (Var, TegVar, Placeholder)):
        return expr

    elif isinstance(expr, TegRemap):
        return TegRemap(
                        map=expr.map,
                        expr=substitute(expr.expr, this_expr, that_expr),
                        exprs={var: substitute(e, this_expr, that_expr) for var, e in expr.exprs.items()},
                        lower_bounds={var: substitute(e, this_expr, that_expr) for var, e in expr.lower_bounds.items()},
                        upper_bounds={var: substitute(e, this_expr, that_expr) for var, e in expr.upper_bounds.items()},
                        source_bounds=expr.source_bounds
                    )

    elif isinstance(expr, Add):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 + simple2

    elif isinstance(expr, Mul):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 * simple2

    elif isinstance(expr, Invert):
        return Invert(substitute(expr.child, this_expr, that_expr))

    elif isinstance(expr, Delta):
        return Delta(substitute(expr.expr, this_expr, that_expr))

    elif isinstance(expr, SmoothFunc):
        return type(expr)(substitute(expr.expr, this_expr, that_expr))

    elif isinstance(expr, IfElse):
        cond = substitute(expr.cond, this_expr, that_expr)
        if_body = substitute(expr.if_body, this_expr, that_expr)
        else_body = substitute(expr.else_body, this_expr, that_expr)
        return IfElse(cond, if_body, else_body)

    elif isinstance(expr, Teg):
        # Ignore shadowed variables
        if expr.dvar == this_expr:
            return expr
        lower = substitute(expr.lower, this_expr, that_expr)
        upper = substitute(expr.upper, this_expr, that_expr)
        body = substitute(expr.body, this_expr, that_expr)
        return Teg(lower, upper, body, expr.dvar)

    elif isinstance(expr, Tup):
        return Tup(*(substitute(child, this_expr, that_expr) for child in expr))

    elif isinstance(expr, LetIn):
        let_expr = substitute(expr.expr, this_expr, that_expr)
        let_body = Tup(*(substitute(e, this_expr, that_expr) for e in expr.new_exprs))
        return LetIn(expr.new_vars, let_body, let_expr)

    elif isinstance(expr, BiMap):
        body_expr = substitute(expr.expr, this_expr, that_expr)
        target_exprs = Tup(*(substitute(e, this_expr, that_expr) for e in expr.target_exprs))
        source_exprs = Tup(*(substitute(e, this_expr, that_expr) for e in expr.source_exprs))
        return BiMap(expr=body_expr,
                     targets=expr.targets,
                     target_exprs=target_exprs,
                     sources=expr.sources,
                     source_exprs=source_exprs)

    elif isinstance(expr, Bool):
        left_expr = substitute(expr.left_expr, this_expr, that_expr)
        right_expr = substitute(expr.right_expr, this_expr, that_expr)
        return Bool(left_expr, right_expr, allow_eq=expr.allow_eq)

    elif isinstance(expr, And):
        left_expr = substitute(expr.left_expr, this_expr, that_expr)
        right_expr = substitute(expr.right_expr, this_expr, that_expr)
        return And(left_expr, right_expr)

    elif isinstance(expr, Or):
        left_expr = substitute(expr.left_expr, this_expr, that_expr)
        right_expr = substitute(expr.right_expr, this_expr, that_expr)
        return Or(left_expr, right_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" is not supported by substitute.')
"""
