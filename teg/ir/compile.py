from functools import reduce
import operator

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    SmoothFunc,
    Tup,
    LetIn,
    Bool,
    And,
    Or,
    Invert,
)
from teg.derivs import FwdDeriv, RevDeriv

from .emit_c import CEmitter


TARGETS = {
    'C': (CEmitter, {}),
    'CUDA_C': (CEmitter, {'device_code': True})
}


def merge(d1: dict, d2: dict):
    new_d = {}
    new_d.update(d1)
    new_d.update(d2)
    return new_d


def clean_name(varname):
    return varname


def emit(expr: ITeg, num_samples: int = 50, ignore_cache: bool = False, target='C',
         fn_name='teg_program', float_type='float', types={}, arglist=[]):
    assert target in TARGETS.keys(), f'Target language "{target}" is not supported at this time.'
    emitter_class, emitter_args = TARGETS[target]
    emitter = emitter_class(float_type=float_type, **emitter_args)

    method = emitter.method(_emit(expr, num_samples, ignore_cache, emitter, type_ctx=types),
                            method_name=fn_name, arglist=arglist)

    return method


def _emit(expr: ITeg, num_samples: int = 50, ignore_cache: bool = False, emitter=None, type_ctx=None):
    assert emitter is not None, 'Use emit() instead of _emit()'

    if isinstance(expr, Const):
        return emitter.literal(expr.value)

    elif isinstance(expr, Var):
        # Lookup type binding in the context.
        if expr.value is None:
            assert (expr.name, expr.uid) in type_ctx.keys(), f'"{expr.name}"" does not have a type binding.'
            ctype, size = type_ctx[(expr.name, expr.uid)]
        else:
            # print(type(expr.value))
            ctype = emitter.float_type
            size = 1
            assert ctype is not None, "Default values other than float or integer are not supported"

        code = emitter.variable(f"{expr.name}_{expr.uid}", ctype=ctype, size=size, assigned=False,
                                default=float(expr.value) if expr.value is not None else None)
        return code

    elif isinstance(expr, (Add, Mul)):
        assert len(expr.children) == 2, f'Add expression "{expr.name}" has an invalid number of operands'

        code1 = _emit(expr.children[0], num_samples, ignore_cache, emitter, type_ctx)
        code2 = _emit(expr.children[1], num_samples, ignore_cache, emitter, type_ctx)

        if isinstance(expr, Add):
            code = emitter.add(code1, code2)
            return code
        elif isinstance(expr, Mul):
            code = emitter.mul(code1, code2)
            return code

    elif isinstance(expr, Invert):
        code = _emit(expr.child, num_samples, ignore_cache, emitter, type_ctx)
        invcode = emitter.invert(code)
        return invcode

    elif isinstance(expr, SmoothFunc):
        code = _emit(expr.expr, num_samples, ignore_cache, emitter, type_ctx)
        fn_code = emitter.smooth_func(code, op=type(expr))
        return fn_code

    elif isinstance(expr, IfElse):
        cond = _emit(expr.cond,      num_samples, ignore_cache, emitter, type_ctx)
        if_body = _emit(expr.if_body,   num_samples, ignore_cache, emitter, type_ctx)
        else_body = _emit(expr.else_body, num_samples, ignore_cache, emitter, type_ctx)

        return emitter.condition(cond, if_body, else_body)

    elif isinstance(expr, Teg):
        lower = _emit(expr.lower, num_samples, ignore_cache, emitter, type_ctx)
        upper = _emit(expr.upper, num_samples, ignore_cache, emitter, type_ctx)

        new_bindings = {(expr.dvar.name, expr.dvar.uid): (lower.out_var.ctype, lower.out_var.size)}
        type_ctx = merge(type_ctx, new_bindings)

        loop_body = _emit(expr.body, num_samples, ignore_cache, emitter, type_ctx)

        ctype, size = new_bindings[(expr.dvar.name, expr.dvar.uid)]
        bind_var = emitter.variable(f"{expr.dvar.name}_{expr.dvar.uid}", ctype=ctype, size=size)

        aggregate_value = emitter.aggregate(
                            emitter.variable().out_var,
                            lower,
                            upper,
                            emitter.literal(num_samples),
                            bind_var.out_var,
                            loop_body
                        )

        return aggregate_value

    elif isinstance(expr, Tup):
        expr_list = [_emit(e, num_samples, ignore_cache, emitter, type_ctx) for e in expr]
        return emitter.array_assign(emitter.variable().out_var,
                                    expr_list)

    elif isinstance(expr, LetIn):
        assignment_list = []
        new_bindings = {}
        for var, e in zip(expr.new_vars, expr.new_exprs):
            var_code = _emit(e, num_samples, ignore_cache, emitter, type_ctx)
            assignment_list.append(
                    emitter.assign(
                        emitter.variable(f"{var.name}_{var.uid}",
                                         ctype=var_code.out_var.ctype,
                                         size=var_code.out_var.size
                                         ).out_var,
                        var_code
                    )
                )
            new_bindings[(var.name, var.uid)] = (var_code.out_var.ctype, var_code.out_var.size)

        assignment_code = reduce(operator.add, assignment_list)

        expr_code = _emit(expr.expr, num_samples, ignore_cache, emitter, merge(type_ctx, new_bindings))

        final_assign_code = emitter.assign(emitter.variable().out_var, expr_code)
        return emitter.block(assignment_code + final_assign_code)

    elif isinstance(expr, (FwdDeriv, RevDeriv)):
        return _emit(expr.deriv_expr, num_samples, ignore_cache, emitter, type_ctx)

    elif isinstance(expr, Bool):
        left_code = _emit(expr.left_expr, num_samples, ignore_cache, emitter, type_ctx)
        right_code = _emit(expr.right_expr, num_samples, ignore_cache, emitter, type_ctx)
        return emitter.compare_lt(left_code, right_code)

    elif isinstance(expr, And):
        left_code = _emit(expr.left_expr, num_samples, ignore_cache, emitter, type_ctx)
        right_code = _emit(expr.right_expr, num_samples, ignore_cache, emitter, type_ctx)
        return emitter.logical_and(left_code, right_code)

    elif isinstance(expr, Or):
        left_code = _emit(expr.left_expr, num_samples, ignore_cache, emitter, type_ctx)
        right_code = _emit(expr.right_expr, num_samples, ignore_cache, emitter, type_ctx)
        return emitter.logical_or(left_code, right_code)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')

    return expr.value
