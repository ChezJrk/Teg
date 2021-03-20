from teg import ITeg
from .base import base_pass


def substitute(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    def inner_fn(e, ctx):
        return e, ctx

    def outer_fn(e, ctx):
        if e == this_expr:
            return that_expr, ctx
        else:
            return e, ctx

    def context_combine(_, ctx):
        return ctx

    n_expr, _ = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return n_expr


def substitute_instance(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    """Substitutes a specific instance of expr (ignores exprs that are symbolically equal to expr but not the same). """
    def inner_fn(e, ctx):
        if e is this_expr:
            return that_expr, ctx
        else:
            return e, ctx

    def outer_fn(e, ctx):
        return e, ctx

    def context_combine(_, ctx):
        return ctx

    n_expr, _ = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return n_expr
