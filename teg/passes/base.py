from typing import Tuple, Dict
from teg import (
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
    Placeholder,
    SmoothFunc,
    TegVar,
    Bool,
    And,
    Or
)

from teg.lang.extended import (
    BiMap,
    Delta
)


def base_pass(expr: ITeg, context, inner_fn, outer_fn, context_combine) -> Tuple[ITeg, Dict]:
    return do_pass(*inner_fn(expr, context), inner_fn, outer_fn, context_combine)


def do_pass(expr: ITeg, context, inner_fn, outer_fn, context_combine) -> Tuple[ITeg, Dict]:
    """Substitute this_var with that_var in expr."""

    if isinstance(expr, Const):
        expr, out_context = outer_fn(expr, context_combine([], context))
        return expr, out_context

    elif isinstance(expr, (Var, TegVar, Placeholder)):
        expr, out_context = outer_fn(expr, context_combine([], context))
        return expr, out_context

    elif isinstance(expr, Add):
        left_expr, left_ctx = do_pass(*inner_fn(expr.children[0], context), inner_fn, outer_fn, context_combine)
        right_expr, right_ctx = do_pass(*inner_fn(expr.children[1], context), inner_fn, outer_fn, context_combine)

        return outer_fn(expr if (left_expr is expr.children[0]) and (right_expr is expr.children[1])
                        else left_expr + right_expr,
                        context_combine([left_ctx, right_ctx], context))

    elif isinstance(expr, Mul):
        left_expr, left_ctx = do_pass(*inner_fn(expr.children[0], context), inner_fn, outer_fn, context_combine)
        right_expr, right_ctx = do_pass(*inner_fn(expr.children[1], context), inner_fn, outer_fn, context_combine)

        return outer_fn(expr if (left_expr is expr.children[0]) and (right_expr is expr.children[1])
                        else left_expr * right_expr,
                        context_combine([left_ctx, right_ctx], context))

    elif isinstance(expr, Invert):
        child, child_ctx = do_pass(*inner_fn(expr.child, context), inner_fn, outer_fn, context_combine)

        return outer_fn(Invert(child) if expr.child is not child else expr, context_combine([child_ctx], context))

    elif isinstance(expr, SmoothFunc):
        child, child_ctx = do_pass(*inner_fn(expr.expr, context), inner_fn, outer_fn, context_combine)

        return outer_fn(type(expr)(child) if expr.expr is not child else expr, context_combine([child_ctx], context))

    elif isinstance(expr, IfElse):
        cond, cond_ctx = do_pass(*inner_fn(expr.cond, context), inner_fn, outer_fn, context_combine)
        if_body, if_ctx = do_pass(*inner_fn(expr.if_body, context), inner_fn, outer_fn, context_combine)
        else_body, else_ctx = do_pass(*inner_fn(expr.else_body, context), inner_fn, outer_fn, context_combine)
        expr = IfElse(cond, if_body, else_body) if (cond is not expr.cond or
                                                    if_body is not expr.if_body or
                                                    else_body is not expr.else_body) else expr

        return outer_fn(expr, context_combine([cond_ctx, if_ctx, else_ctx], context))

    elif isinstance(expr, Teg):
        # dvar, dvar_ctx = do_pass(*inner_fn(expr.dvar, context), inner_fn, outer_fn, context_combine)
        body, body_ctx = do_pass(*inner_fn(expr.body, context), inner_fn, outer_fn, context_combine)
        lower, lower_ctx = do_pass(*inner_fn(expr.lower, context), inner_fn, outer_fn, context_combine)
        upper, upper_ctx = do_pass(*inner_fn(expr.upper, context), inner_fn, outer_fn, context_combine)

        expr = expr if (body is expr.body and
                        lower is expr.lower and
                        upper is expr.upper) else Teg(lower, upper, body, expr.dvar)

        return outer_fn(expr, context_combine([lower_ctx, upper_ctx, body_ctx], context))

    elif isinstance(expr, Tup):
        exprs, expr_contexts = zip(*[(do_pass(*inner_fn(child, context), inner_fn, outer_fn, context_combine))
                                     for child in expr])
        expr = expr if all([new_child is old_child for new_child, old_child in zip(exprs, expr)]) else Tup(*exprs)

        return outer_fn(expr, context_combine(expr_contexts, context))

    elif isinstance(expr, LetIn):
        body_expr, body_context = do_pass(*inner_fn(expr.expr, context), inner_fn, outer_fn, context_combine)
        let_exprs, let_contexts = zip(*[do_pass(*inner_fn(child, context), inner_fn, outer_fn, context_combine)
                                        for child in expr.new_exprs])
        expr = expr if (all([new_child is old_child for new_child, old_child in zip(let_exprs, expr.new_exprs)])
                        and body_expr is expr.expr) else LetIn(expr.new_vars, let_exprs, body_expr)

        return outer_fn(expr, context_combine([body_context, *let_contexts], context))

    elif isinstance(expr, Bool):
        left_expr, left_ctx = do_pass(*inner_fn(expr.left_expr, context), inner_fn, outer_fn, context_combine)
        right_expr, right_ctx = do_pass(*inner_fn(expr.right_expr, context), inner_fn, outer_fn, context_combine)

        expr = expr if (left_expr is expr.left_expr and
                        right_expr is expr.right_expr
                        ) else Bool(left_expr, right_expr, allow_eq=expr.allow_eq)
        return outer_fn(expr,
                        context_combine([left_ctx, right_ctx], context))

    elif isinstance(expr, (And, Or)):
        left_expr, left_ctx = do_pass(*inner_fn(expr.left_expr, context), inner_fn, outer_fn, context_combine)
        right_expr, right_ctx = do_pass(*inner_fn(expr.right_expr, context), inner_fn, outer_fn, context_combine)

        expr = expr if (left_expr is expr.left_expr and
                        right_expr is expr.right_expr
                        ) else type(expr)(left_expr, right_expr)

        return outer_fn(expr,
                        context_combine([left_ctx, right_ctx], context))

    elif isinstance(expr, BiMap):
        body_expr, body_context = do_pass(*inner_fn(expr.expr, context), inner_fn, outer_fn, context_combine)
        source_exprs, source_contexts = zip(*[do_pass(*inner_fn(child, context), inner_fn, outer_fn, context_combine)
                                              for child in expr.source_exprs])
        target_exprs, target_contexts = zip(*[do_pass(*inner_fn(child, context), inner_fn, outer_fn, context_combine)
                                              for child in expr.target_exprs])
        jacobian_expr, jacobian_context = do_pass(*inner_fn(expr.inv_jacobian, context), inner_fn, outer_fn,
                                                  context_combine)
        target_upper_bounds, ub_contexts = zip(*[do_pass(*inner_fn(child, context), inner_fn, outer_fn,
                                                         context_combine)
                                                 for child in expr.target_upper_bounds])
        target_lower_bounds, lb_contexts = zip(*[do_pass(*inner_fn(child, context), inner_fn, outer_fn,
                                                         context_combine)
                                                 for child in expr.target_lower_bounds])

        expr = expr if (all([new_child is old_child for new_child, old_child in
                             zip(source_exprs, expr.source_exprs)]) and
                        all([new_child is old_child for new_child, old_child in
                             zip(target_exprs, expr.target_exprs)]) and
                        all([new_child is old_child for new_child, old_child in
                             zip(target_upper_bounds, expr.target_upper_bounds)]) and
                        all([new_child is old_child for new_child, old_child in
                             zip(target_lower_bounds, expr.target_lower_bounds)]) and
                        body_expr is expr.expr and
                        jacobian_expr is expr.inv_jacobian) else BiMap(body_expr,
                                                                   expr.targets,
                                                                   target_exprs,
                                                                   expr.sources,
                                                                   source_exprs,
                                                                   jacobian_expr,
                                                                   target_upper_bounds,
                                                                   target_lower_bounds)

        return outer_fn(expr, context_combine([*source_contexts, *target_contexts,
                                               jacobian_context,
                                               *ub_contexts, *lb_contexts,
                                               body_context], context))

    elif isinstance(expr, Delta):
        body_expr, body_context = do_pass(*inner_fn(expr.expr, context), inner_fn, outer_fn, context_combine)

        expr = expr if body_expr is expr.expr else Delta(body_expr)

        return outer_fn(expr, context_combine([body_context], context))

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" is not supported by substitute.')
