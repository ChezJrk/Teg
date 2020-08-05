from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Invert,
    IfElse,
    Teg,
    Tup,
    LetIn,
    Or,
    And,
    Bool,
    true,
    false,
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

    elif isinstance(expr, Invert):
        return Invert(simplify(expr.child))

    elif isinstance(expr, IfElse):
        cond, if_body, else_body = simplify(expr.cond), simplify(expr.if_body), simplify(expr.else_body)
        if (isinstance(if_body, Const) and isinstance(else_body, Const)
                and if_body.value == 0 and else_body.value == 0):
            return if_body

        if cond == true:
            return if_body

        if cond == false:
            return else_body

        return IfElse(cond, if_body, else_body)

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

    elif isinstance(expr, Bool):
        return Bool(simplify(expr.left_expr), simplify(expr.right_expr))

    elif isinstance(expr, And):
        left_expr, right_expr = simplify(expr.left_expr), simplify(expr.right_expr)
        if left_expr == true:
            return right_expr
        if right_expr == true:
            return left_expr
        if left_expr == false or right_expr == false:
            return false
        return And(left_expr, right_expr)

    elif isinstance(expr, Or):
        left_expr, right_expr = simplify(expr.left_expr), simplify(expr.right_expr)
        if left_expr == false:
            return right_expr
        if right_expr == false:
            return left_expr
        if left_expr == true or right_expr == true:
            return true
        return Or(left_expr, right_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')
