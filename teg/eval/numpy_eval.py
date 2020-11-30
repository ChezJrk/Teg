import numpy as np

from teg import (
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
    SmoothFunc
)

from teg.derivs import FwdDeriv, RevDeriv


def evaluate(expr: ITeg, num_samples: int = 50, ignore_cache: bool = False):

    if expr.value is not None and not ignore_cache:
        if isinstance(expr, Var):
            assert expr.value is not None, f'The variable "{expr.name}" must be bound to a value prior to evaluation.'
        return expr.value

    if isinstance(expr, (Const, Var)):
        expr.value = expr.value

    elif isinstance(expr, (Add, Mul)):
        expr.value = expr.operation(*[evaluate(e, num_samples, ignore_cache) for e in expr.children])

    elif isinstance(expr, SmoothFunc):
        expr.value = expr.operation(evaluate(expr.expr))

    elif isinstance(expr, Invert):
        expr.value = 1 / evaluate(expr.child, num_samples, ignore_cache)

    elif isinstance(expr, IfElse):
        cond = evaluate(expr.cond, num_samples, ignore_cache)
        body = expr.if_body if cond else expr.else_body
        expr.value = evaluate(body, num_samples, ignore_cache)

    elif isinstance(expr, Teg):
        lower = evaluate(expr.lower, num_samples, ignore_cache)
        upper = evaluate(expr.upper, num_samples, ignore_cache)

        expr.dvar.value = None

        # Sample different values of the variable (dvar) and evaluate
        # Currently do NON-DIFFERENTIABLE uniform sampling
        def compute_samples(var_sample):
            expr.body.bind_variable(expr.dvar, var_sample)
            return evaluate(expr.body, num_samples, ignore_cache=True)

        var_samples, step = np.linspace(lower, upper, num_samples, retstep=True)

        # Hacky way to handle integrals of vectors
        try:
            body_at_samples = np.vectorize(compute_samples, otypes=[np.float])(var_samples)
        except ValueError:
            body_at_samples = np.vectorize(compute_samples, signature='()->(n)')(var_samples)

        # Trapezoidal rule
        y_left = body_at_samples[:-1]  # left endpoints
        y_right = body_at_samples[1:]  # right endpoints
        expr.value = (1 if lower < upper else -1) * step * np.sum(y_left + y_right, 0) / 2

    elif isinstance(expr, Tup):
        expr.value = np.array([evaluate(e, num_samples, ignore_cache) for e in expr])

    elif isinstance(expr, LetIn):
        for var, e in zip(expr.new_vars, expr.new_exprs):
            var_val = evaluate(e, num_samples, ignore_cache)
            expr.expr.bind_variable(var, var_val)
        expr.value = evaluate(expr.expr, num_samples, ignore_cache)

    elif isinstance(expr, (FwdDeriv, RevDeriv)):
        expr.value = evaluate(expr.deriv_expr, num_samples, ignore_cache)

    elif isinstance(expr, Bool):
        left_val = evaluate(expr.left_expr, num_samples, ignore_cache)
        right_val = evaluate(expr.right_expr, num_samples, ignore_cache)
        expr.value = (left_val < right_val) or (expr.allow_eq and (expr.left_val == expr.right_val))

    elif isinstance(expr, And):
        left_val = evaluate(expr.left_expr, num_samples, ignore_cache)
        right_val = evaluate(expr.right_expr, num_samples, ignore_cache)
        expr.value = left_val and right_val

    elif isinstance(expr, Or):
        left_val = evaluate(expr.left_expr, num_samples, ignore_cache)
        right_val = evaluate(expr.right_expr, num_samples, ignore_cache)
        expr.value = left_val or right_val

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')

    assert expr.value is not None, f'Error evaluating expression {expr}'
    return expr.value
