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
)
import operator_overloads  # noqa: F401


def substitute(expr: Teg, this_expr: Teg, that_expr: Teg) -> Teg:
    """Substitute this_var with that_var in expr. """

    if isinstance(expr, TegConstant):
        return expr

    elif expr == this_expr:
        return that_expr

    elif isinstance(expr, TegVariable):
        return expr

    elif isinstance(expr, TegAdd):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 + simple2

    elif isinstance(expr, TegMul):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_expr, that_expr), substitute(expr2, this_expr, that_expr)
        return simple1 * simple2

    elif isinstance(expr, TegConditional):
        lt_expr = substitute(expr.lt_expr, this_expr, that_expr)
        if_body = substitute(expr.if_body, this_expr, that_expr)
        else_body = substitute(expr.else_body, this_expr, that_expr)
        return TegConditional(lt_expr, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, TegIntegral):
        # Ignore shadowed variables
        if expr.dvar == this_expr:
            return expr
        body = substitute(expr.body, this_expr, that_expr)
        return TegIntegral(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, TegTuple):
        return TegTuple(*(substitute(child, this_expr, that_expr) for child in expr))

    elif isinstance(expr, TegLetIn):
        let_expr = substitute(expr.expr, this_expr, that_expr)
        let_body = TegTuple(*(substitute(e, this_expr, that_expr) for e in expr.new_exprs))
        return TegLetIn(expr.new_vars, let_body, let_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
