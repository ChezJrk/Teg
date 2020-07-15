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


def substitute(expr: Teg, this_var: TegVariable, that_var: TegVariable) -> Teg:
    """Substitute this_var with that_var in expr. """

    if isinstance(expr, TegVariable):
        if expr.name == this_var.name:
            return that_var
        return expr

    elif isinstance(expr, TegAdd):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_var, that_var), substitute(expr2, this_var, that_var)
        return simple1 + simple2

    elif isinstance(expr, TegMul):
        expr1, expr2 = expr.children
        simple1, simple2 = substitute(expr1, this_var, that_var), substitute(expr2, this_var, that_var)
        return simple1 * simple2

    elif isinstance(expr, TegConditional):
        v1, v2 = substitute(expr.var1, this_var, that_var), substitute(expr.var2, this_var, that_var)
        if_body = substitute(expr.if_body, this_var, that_var)
        else_body = substitute(expr.else_body, this_var, that_var)
        return TegConditional(v1, v2, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, TegIntegral):
        # Ignore shadowed variables
        if expr.dvar.name == this_var.name:
            return expr
        body = substitute(expr.body, this_var, that_var)
        return TegIntegral(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, TegTuple):
        return TegTuple(*(substitute(child, this_var, that_var) for child in expr))

    elif isinstance(expr, TegLetIn):
        let_expr = substitute(expr.expr, this_var, that_var)
        let_body = TegTuple(*(substitute(e, this_var, that_var) for e in expr.new_exprs))
        return TegLetIn(expr.new_vars, let_body, let_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
