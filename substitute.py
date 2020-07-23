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
import operator_overloads  # noqa: F401


def substitute(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    """Substitute this_var with that_var in expr. """

    if isinstance(expr, Const):
        return expr

    elif expr == this_expr:
        return that_expr

    elif isinstance(expr, Var):
        return expr

    elif isinstance(expr, Add):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 + simple2

    elif isinstance(expr, Mul):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 * simple2

    elif isinstance(expr, Cond):
        lt_expr = substitute(expr.lt_expr, this_expr, that_expr)
        if_body = substitute(expr.if_body, this_expr, that_expr)
        else_body = substitute(expr.else_body, this_expr, that_expr)
        return Cond(lt_expr, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, Teg):
        # Ignore shadowed variables
        if expr.dvar == this_expr:
            return expr
        body = substitute(expr.body, this_expr, that_expr)
        return Teg(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, Tup):
        return Tup(*(substitute(child, this_expr, that_expr) for child in expr))

    elif isinstance(expr, LetIn):
        let_expr = substitute(expr.expr, this_expr, that_expr)
        let_body = Tup(*(substitute(e, this_expr, that_expr) for e in expr.new_exprs))
        return LetIn(expr.new_vars, let_body, let_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
