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
from derivs import FwdDeriv, RevDeriv
import operator_overloads  # noqa: F401


def simplify(expr: ITeg) -> ITeg:

    if isinstance(expr, Var):
        return expr

    elif isinstance(expr, Add):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if isinstance(simple1, Var) and simple1.value == 0:
            return simple2
        if isinstance(simple2, Var) and simple2.value == 0:
            return simple1
        return simple1 + simple2

    elif isinstance(expr, Mul):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if ((isinstance(simple1, Var) and simple1.value == 0)
                or (isinstance(simple2, Var) and hasattr(simple2, 'value') and simple2.value == 0)):
            return Const(0)
        if isinstance(simple1, Var) and simple1.value == 1:
            return simple2
        if isinstance(simple2, Var) and simple2.value == 1:
            return simple1
        return simple1 * simple2

    elif isinstance(expr, Cond):
        lt_expr, if_body, else_body = simplify(expr.lt_expr), simplify(expr.if_body), simplify(expr.else_body)
        if (isinstance(if_body, Const) and isinstance(else_body, Const)
                and if_body.value == 0 and else_body.value == 0):
            return if_body

        # When lt_expr has a value (it's a bound variable or constant)
        # return the appropriate branch
        try:
            if lt_expr.value is not None:
                if lt_expr.value < 0 or (expr.allow_eq and lt_expr.value == 0):
                    return if_body
                else:
                    return else_body
        except AttributeError:
            pass

        return Cond(lt_expr, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, Teg):
        body = simplify(expr.body)
        if isinstance(body, Var) and hasattr(body, 'value') and body.value == 0:
            return Const(0)
        return Teg(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, Tup):
        return Tup(*(simplify(child) for child in expr))

    elif isinstance(expr, LetIn):
        return LetIn(expr.new_vars, Tup(*(simplify(e) for e in expr.new_exprs)), simplify(expr.expr))

    elif isinstance(expr, (FwdDeriv, RevDeriv)):
        return simplify(expr.deriv_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
