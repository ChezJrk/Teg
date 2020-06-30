import numpy as np

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


def evaluate(expr: Teg, num_samples: int = 50, ignore_cache: bool = False):

    if expr.value is not None and not ignore_cache:
        if isinstance(expr, TegVariable):
            assert expr.value is not None, f'The variable "{expr.name}" must be bound to a value prior to evaluation.'
        return expr.value * expr.sign

    if isinstance(expr, (TegConstant, TegVariable)):
        expr.value = expr.value

    elif isinstance(expr, (TegAdd, TegMul)):
        expr.value = expr.operation(*[evaluate(e, num_samples, ignore_cache) for e in expr.children])

    elif isinstance(expr, TegConditional):
        expr.value = evaluate(expr.if_body if expr.var < expr.const else expr.else_body, num_samples, ignore_cache)

    elif isinstance(expr, TegIntegral):
        lower = evaluate(expr.lower, num_samples, ignore_cache)
        upper = evaluate(expr.upper, num_samples, ignore_cache)
        expr.sign *= (1 if lower < upper else -1)

        expr.dvar.value = None

        # Sample different values of the variable (dvar) and evaluate
        # Currently do NON-DIFFERENTIABLE uniform sampling
        def compute_samples(var_sample):
            expr.body.bind_variable(expr.dvar.name, var_sample)
            return evaluate(expr.body, num_samples, ignore_cache=True)

        var_samples, step = np.linspace(lower, upper, num_samples, retstep=True)

        # Hacky way to handle integrals of vectors
        try:
            body_at_samples = np.vectorize(compute_samples)(var_samples)
        except ValueError:
            body_at_samples = np.vectorize(compute_samples, signature='()->(n)')(var_samples)

        # Trapezoidal rule
        y_left = body_at_samples[:-1]  # left endpoints
        y_right = body_at_samples[1:]  # right endpoints
        expr.value = step * np.sum(y_left + y_right, 0) / 2

    elif isinstance(expr, TegTuple):
        expr.value = np.array([evaluate(e, num_samples, ignore_cache) for e in expr])

    elif isinstance(expr, TegLetIn):
        for var, e in zip(expr.new_vars, expr.new_exprs):
            var_val = evaluate(e, num_samples, ignore_cache)
            expr.expr.bind_variable(var.name, var_val)
        expr.value = evaluate(expr.expr, num_samples, ignore_cache)

    elif isinstance(expr, (TegFwdDeriv, TegReverseDeriv)):
        expr.value = evaluate(expr.deriv_expr, num_samples, ignore_cache)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')

    return expr.value