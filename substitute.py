from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    Tup,
    LetIn,
    Bool,
    And,
    Or,
    Invert,
)
import operator_overloads  # noqa: F401


def substitute(expr: ITeg, this_expr: ITeg, that_expr: ITeg) -> ITeg:
    """Substitute this_var with that_var in expr. """

    if isinstance(expr, Const):
        return expr

    elif (isinstance(expr, Var) or isinstance(expr, Bool)) and expr == this_expr:
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

    elif isinstance(expr, Invert):
        return Invert(substitute(expr.child, this_expr, that_expr))

    elif isinstance(expr, IfElse):
        cond = substitute(expr.cond, this_expr, that_expr)
        if_body = substitute(expr.if_body, this_expr, that_expr)
        else_body = substitute(expr.else_body, this_expr, that_expr)
        return IfElse(cond, if_body, else_body)

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

    elif isinstance(expr, Bool):
        left_expr = substitute(expr.left_expr, this_expr, that_expr)
        right_expr = substitute(expr.right_expr, this_expr, that_expr)
        return Bool(left_expr, right_expr)

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
