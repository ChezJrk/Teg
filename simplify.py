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
from derivs import TegFwdDeriv, TegReverseDeriv
import operator_overloads  # noqa: F401


def simplify(expr: Teg) -> Teg:

    if isinstance(expr, TegVariable):
        return expr

    elif isinstance(expr, TegAdd):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if isinstance(simple1, TegVariable) and simple1.value == 0:
            return simple2
        if isinstance(simple2, TegVariable) and simple2.value == 0:
            return simple1
        return simple1 + simple2

    elif isinstance(expr, TegMul):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if ((isinstance(simple1, TegVariable) and simple1.value == 0)
                or (isinstance(simple2, TegVariable) and hasattr(simple2, 'value') and simple2.value == 0)):
            return TegConstant(0)
        if isinstance(simple1, TegVariable) and simple1.value == 1:
            return simple2
        if isinstance(simple2, TegVariable) and simple2.value == 1:
            return simple1
        return simple1 * simple2

    elif isinstance(expr, TegConditional):
        lt_expr, if_body, else_body = simplify(expr.lt_expr), simplify(expr.if_body), simplify(expr.else_body)
        if (isinstance(if_body, TegConstant) and isinstance(else_body, TegConstant)
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

        return TegConditional(lt_expr, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, TegIntegral):
        body = simplify(expr.body)
        if isinstance(body, TegVariable) and hasattr(body, 'value') and body.value == 0:
            return TegConstant(0)
        return TegIntegral(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, TegTuple):
        return TegTuple(*(simplify(child) for child in expr))

    elif isinstance(expr, TegLetIn):
        return TegLetIn(expr.new_vars, TegTuple(*(simplify(e) for e in expr.new_exprs)), simplify(expr.expr))

    elif isinstance(expr, (TegFwdDeriv, TegReverseDeriv)):
        return simplify(expr.deriv_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
