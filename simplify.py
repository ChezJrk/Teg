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
        v1, v2, if_body, else_body = expr.var1, expr.var2, simplify(expr.if_body), simplify(expr.else_body)
        if (isinstance(v1, TegVariable) and isinstance(v2, TegVariable)
                and v1.value is not None and v2.value is not None):
            if v1.value < v2.value or (expr.allow_eq and v1.value == v2.value):
                return if_body
            else:
                return else_body
        return TegConditional(expr.var1, expr.var2, if_body, else_body, allow_eq=expr.allow_eq)

    elif isinstance(expr, TegIntegral):
        body = simplify(expr.body)
        if isinstance(body, TegVariable) and hasattr(body, 'value') and body.value == 0:
            return TegConstant(0)
        return TegIntegral(expr.lower, expr.upper, body, expr.dvar)

    elif isinstance(expr, TegTuple):
        if len(expr.children) == 1:
            return simplify(expr.children[0])
        return TegTuple(*(simplify(child) for child in expr))

    elif isinstance(expr, TegLetIn):
        return TegLetIn(expr.new_vars, TegTuple(*(simplify(e) for e in expr.new_exprs)), simplify(expr.expr))

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
