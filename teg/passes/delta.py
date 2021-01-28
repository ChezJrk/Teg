# Delta and BiMap elimination passes
from functools import reduce
from typing import List
from teg.passes.base import base_pass
from teg.passes.substitute import substitute, substitute_instance
from teg.passes.simplify import simplify
from teg import (
    ITeg,
    Add,
    SmoothFunc,
    Teg,
    TegVar,
    Const,
    LetIn,
    IfElse,
    Tup,
    Var,
    Const
)

from teg.lang.extended import (
    Delta, BiMap
)
from teg.lang.extended_utils import (
    top_level_instance_of,
    is_delta_normal,
    resolve_placeholders
)

from teg.derivs.edge.handlers import (
    bilinear,
    affine,
    single_axis
)

import operator
from copy import copy

# In order of precedence..
HANDLERS = [
    single_axis.ConstantAxisHandler,
    affine.AffineHandler,
    bilinear.BilinearHandler
]


def tree_copy(expr: ITeg):
    def inner_fn(e, ctx):
        return e, ctx

    def outer_fn(e, ctx):
        if isinstance(e, Var):
            # Duplicate leaf nodes to force the tree to be
            # freshly constructed. (Prevents duplicate nodes)
            e = copy(e)
        return e, ctx

    def context_combine(ctxs, ctx):
        return ctx

    expr, _ = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return expr


def split_expr(expr: ITeg, t_expr: ITeg):
    return split_exprs([expr], t_expr)


def split_exprs(exprs: List[ITeg], t_expr: ITeg):
    """
    Given a list of expressions exprs in a tree t_expr, find the normalized expression tree
    n_expr such that t_expr = (let exprs = 0 in t_expr) + n_expr
    (expr is linear in t_expr)
    """
    # print(expr)
    # print('\n\n')
    # print(t_expr)
    # print('\n\n')

    def inner_fn(e, ctx):
        ctx = {'expr': e, 'is_expr': e in exprs}
        if isinstance(e, LetIn):
            ctx['let_body'] = e.expr
        return e, ctx

    def outer_fn(e, ctx):
        # if isinstance(e, Var):
        # Duplicate leaf nodes to force the tree to be 
        # freshly constructed. (Prevents duplicate nodes)
        # e = copy(e)

        ctx['has_expr'] = any(ctx['has_exprs'])
        # Check if we need to handle other such cases.
        assert not (ctx['has_expr'] and isinstance(e, SmoothFunc)),\
               f'expr is contained in a non-linear function {type(e)}'
        if isinstance(e, Add):
            if ctx['has_expr']:
                # print('FOUND ADD. Taking branch')
                # print(ctx['has_exprs'])
                ctx['expr'] = sum([child for child, has_expr in zip(ctx['exprs'], ctx['has_exprs']) if has_expr])
                return ctx['expr'], ctx
            else:
                ctx['expr'] = e
                return e, ctx
        elif isinstance(e, Tup):
            if ctx['has_expr']:
                # assert sum(ctx['has_exprs']) == 1, f'More than one branch with expr'
                ctx['expr'] = Tup(*[ctx['exprs'][idx] if has_expr else Const(0)
                                    for idx, has_expr in enumerate(ctx['has_exprs'])])
                return ctx['expr'], ctx
        elif isinstance(e, IfElse):
            if ctx['has_expr']:
                ctx['expr'] = IfElse(e.cond,
                                     ctx['exprs'][1] if ctx['has_exprs'][1] else Const(0),
                                     ctx['exprs'][2] if ctx['has_exprs'][2] else Const(0))
                return ctx, e
        elif isinstance(e, LetIn):
            if any(ctx['has_exprs'][1:]):
                # Let expressions contain exprs.
                new_exprs = [let_var for let_var, has_expr in zip(e.new_vars, ctx['has_exprs'][1:]) if has_expr]
                # Recursively split the body with the new expressions.
                # print('Recursing on: ', new_exprs, ' ', ctx['has_exprs'], ' ON ', ctx['let_body'])
                s_expr = split_exprs(new_exprs, ctx['let_body'])
                # print(f'GOT: {s_expr}')
                # print(f'EXIST: {e.expr}')
                let_body = (s_expr if s_expr else Const(0)) +\
                           (e.expr if ctx['has_exprs'][0] else Const(0))
                try:
                    vs, es = zip(*[(v, e) for v, e in zip(e.new_vars, e.new_exprs) if v in let_body])
                    ctx['expr'] = LetIn(vs, es,
                                        let_body)
                except ValueError:
                    # No need for a let expr.
                    ctx['expr'] = let_body

                return ctx['expr'], ctx

        # print('AT:', e, ' HAS_EXPRS: ', ctx['has_exprs'], 'IS_EXPR: ', ctx.get('is_expr', False))
        ctx['expr'] = e
        return ctx['expr'], ctx

    def context_combine(contexts, ctx):
        return {**ctx,
                'exprs': [context['expr'] for context in contexts],
                'has_exprs': [context.get('has_expr', False) or context.get('is_expr', False) for context in contexts],
                }

    n_expr, context = base_pass(t_expr, {}, inner_fn, outer_fn, context_combine)
    if context['has_expr'] or context['is_expr']:
        return n_expr
    else:
        # Didn't find expr in tree.
        return None


def split_instance(expr: ITeg, t_expr: ITeg):
    """
    Given a specific expression instance expr in a tree t_expr, find the normalized expression tree
    n_expr such that t_expr = (let_instance expr = 0 in t_expr) + n_expr
    (expr is linear in t_expr)
    """

    def inner_fn(e, ctx):
        ctx = {'expr': e, 'is_expr': expr is e}
        if isinstance(e, LetIn):
            ctx['let_body'] = e.expr
        return e, ctx

    def outer_fn(e, ctx):
        # if isinstance(e, Var):
        # Duplicate leaf nodes to force the tree to be
        # freshly constructed. (Prevents duplicate nodes)
        # e = copy(e)

        ctx['has_expr'] = any(ctx['has_exprs'])
        # Check if we need to handle other such cases.
        assert not (ctx['has_expr'] and isinstance(e, SmoothFunc)),\
               f'expr is contained in a non-linear function {type(e)}'
        if isinstance(e, Add):
            if ctx['has_expr']:

                assert sum(ctx['has_exprs']) == 1, 'More than one branch with expr'
                ctx['expr'] = ctx['exprs'][ctx['has_exprs'].index(True)]
                return ctx['expr'], ctx
            else:
                ctx['expr'] = e
                return e, ctx
        elif isinstance(e, IfElse):
            if ctx['has_expr']:
                assert sum(ctx['has_exprs']) == 1, 'More than one branch with expr'
                if ctx['has_exprs'][1]:
                    # If block contains expr.
                    ctx['expr'] = IfElse(e.cond, ctx['exprs'][1], Const(0))
                elif ctx['has_exprs'][2]:
                    ctx['expr'] = IfElse(e.cond, Const(0), ctx['exprs'][2])
                else:
                    assert False, 'condition must not contain expr. expr is not linear'

                return ctx['expr'], ctx
        elif isinstance(e, Tup):
            if ctx['has_expr']:
                assert sum(ctx['has_exprs']) == 1, 'More than one branch with expr'
                ctx['expr'] = Tup(*[ctx['exprs'][idx] if has_expr else Const(0)
                                    for idx, has_expr in enumerate(ctx['has_exprs'])])
                return ctx['expr'], ctx
        elif isinstance(e, LetIn):
            assert sum(ctx['has_exprs']) <= 1, 'More than one branch with expr'
            if any(ctx['has_exprs'][1:]):
                # Let expressions contain exprs.
                new_exprs = [let_var for let_var, has_expr in zip(e.new_vars, ctx['has_exprs'][1:]) if has_expr]

                # Recursively split the body with the new expressions.
                s_expr = split_exprs(new_exprs, ctx['let_body'])
                let_body = (s_expr if s_expr else Const(0)) +\
                           (e.expr if ctx['has_exprs'][0] else Const(0))
                try:
                    vs, es = zip(*[(v, e) for v, e in zip(e.new_vars, e.new_exprs) if v in let_body])
                    ctx['expr'] = LetIn(vs, es,
                                        let_body)
                except ValueError:
                    # No need for a let expr.
                    ctx['expr'] = let_body

                return ctx['expr'], ctx

        ctx['expr'] = e
        return ctx['expr'], ctx

    def context_combine(contexts, ctx):
        return {**ctx,
                'exprs': [context['expr'] for context in contexts],
                'has_exprs': [context.get('has_expr', False) or context.get('is_expr', False) for context in contexts],
                }

    n_expr, context = base_pass(t_expr, {}, inner_fn, outer_fn, context_combine)
    if context['has_expr']:
        return n_expr
    else:
        # Didn't find expr in tree.
        return None


def is_expr_linear_in_tree(expr: ITeg, t_expr: ITeg):
    """
    Given an expression expr in a tree t_expr, find the normalized expression tree
    n_expr such that t_expr = (let expr = 0 in t_expr) + n_expr
    (expr is linear in t_expr)
    """
    def inner_fn(e, ctx):
        return e, {'expr': e, 'is_expr': expr is e, **ctx}

    def outer_fn(e, ctx):
        ctx['has_expr'] = any(ctx['has_exprs'])
        # Check if we need to handle other such cases.
        if ctx['has_expr'] and isinstance(e, (Add, SmoothFunc)):
            ctx['is_linear'] = False
            return e, ctx
        else:
            ctx['is_linear'] = True and all(ctx['is_linears'])
            return e, ctx

    def context_combine(contexts, ctx):
        return {'exprs': [ctx['expr'] for ctx in contexts],
                'has_exprs': [ctx.get('has_expr', False) or ctx.get('is_expr', False) for ctx in contexts],
                'is_linears': [ctx.get('is_linear', True) for ctx in contexts]}

    n_expr, context = base_pass(expr, {}, inner_fn, outer_fn, context_combine)
    return context['is_linear']


def normalize_deltas(expr: Delta):
    # Two cases:
    # 1. Delta is not dependent on its integrals. Set to 0
    # 2. Delta is dependent on its integrals either directly or through a map.
    #    Convert to normal form using handlers. Error if no handler exists.
    def inner_fn(e, ctx):
        if isinstance(e, Teg):
            return e, {'is_expr': expr is e, 'upper_tegvars': ctx['upper_tegvars'] | {e.dvar},
                       'upper_depvars': ctx['upper_depvars'] - {e.dvar},
                       'lower_tegvars': set()}
        elif isinstance(e, BiMap):
            return e, {'is_expr': expr is e, 'upper_tegvars': ctx['upper_tegvars'] | set(e.sources) | set(e.targets),
                       'upper_depvars': (ctx['upper_depvars'] - set(e.sources)) - set(e.targets),
                       'lower_tegvars': set()}
        elif isinstance(e, LetIn):
            return e, {'is_expr': expr is e, 'upper_tegvars': ctx['upper_tegvars'],
                       'upper_depvars': (ctx['upper_depvars'] |
                                         {nvar for nvar, nexpr in zip(e.new_vars, e.new_exprs)
                                          if any(tegvar in nexpr for tegvar in
                                          (ctx['upper_tegvars'] | ctx['upper_depvars']))}),
                       'lower_tegvars': set()}
        else:
            return e, {'is_expr': expr is e, 'upper_tegvars': ctx['upper_tegvars'],
                       'upper_depvars': ctx['upper_depvars'],
                       'lower_tegvars': set()}

    def outer_fn(e, ctx):
        if isinstance(e, Delta):
            depvars = list(tegvar for tegvar in (ctx['upper_depvars'] - ctx['upper_tegvars']) if tegvar in e)
            assert not depvars,\
                   f'Delta expression {e} is not explicitly affine: ({depvars}) '\
                   f'is/are dependent on one or more of {ctx["upper_tegvars"]} '\
                   f'through one-way let expressions. Use bijective maps (BiMap) instead'
            if (not any([k in ctx['upper_tegvars'] for k in ctx['lower_tegvars']])) or (not ctx['lower_tegvars']):
                return Const(0), ctx
            else:
                if not is_delta_normal(e):
                    accepts = [handler.accept(e, set(ctx['upper_tegvars'])) for handler in HANDLERS]
                    assert any(accepts), f'Cannot find any handler for delta expression {e}'

                    handler = HANDLERS[accepts.index(True)]
                    e = handler.rewrite(e, set(ctx['upper_tegvars']))
                    e = normalize_deltas(e)  # Normalize further if necessary

                return e, ctx

        elif isinstance(e, BiMap):
            return e, {**ctx, 'lower_tegvars': ctx['lower_tegvars'] - set(e.targets)}

        elif isinstance(e, TegVar):
            return e, {**ctx, 'lower_tegvars': ctx['lower_tegvars'] | {e}}

        return e, ctx

    def context_combine(contexts, ctx):
        return {'lower_tegvars': reduce(lambda a, b: a | b, [ctx['lower_tegvars'] for ctx in contexts], set()),
                'upper_tegvars': ctx['upper_tegvars'],
                'upper_depvars': ctx['upper_depvars']}

    n_expr, context = base_pass(expr, {'upper_tegvars': set(), 'upper_depvars': set()},
                                inner_fn, outer_fn, context_combine)
    return n_expr


def reparameterize(bimap: BiMap, expr: ITeg):
    # find bimap and all superseding integrals
    # substitute for integrals, generate let exprs, multiply inv_jacobian

    def inner_fn(e, ctx):
        if isinstance(e, Teg):
            return e, {'is_expr': bimap is e,
                       'upper_tegvars': ctx['upper_tegvars'] | {e.dvar},
                       'source_bounds': {**ctx.get('source_bounds', {}), e.dvar: (e.lower, e.upper)}}
        elif isinstance(e, BiMap):
            return e, {'is_expr': bimap is e,
                       'source_bounds': ctx['source_bounds'],
                       'upper_tegvars': ctx['upper_tegvars'] | set(e.targets)}
        else:
            return e, {'is_expr': bimap is e,
                       'upper_tegvars': ctx['upper_tegvars'],
                       'source_bounds': ctx['source_bounds']}

    def outer_fn(e, ctx):
        if isinstance(e, BiMap) and (bimap is e):
            if not all([k in ctx['upper_tegvars'] for k in e.sources]):
                # BiMap is invalid, null everything.
                print(f'WARNING: Attempting to map non-Teg vars {e.sources}, {ctx["upper_tegvars"]}')
                return Const(0), ctx

            bounds_checks = reduce(operator.and_,
                                   [(lb < dvar) & (ub > dvar) for (dvar, (lb, ub)) in ctx['source_bounds'].items()])
            reparamaterized_expr = IfElse(bounds_checks, e.expr * e.inv_jacobian, Const(0))

            return (reparamaterized_expr,
                    {**ctx,
                     'teg_sources': list(e.sources),
                     'teg_targets': list(e.targets),
                     'let_mappings': {s: sexpr for s, sexpr in zip(e.sources, e.source_exprs)},
                     'target_lower_bounds': {t: tlb for t, tlb in zip(e.targets, e.target_lower_bounds)},
                     'target_upper_bounds': {t: tub for t, tub in zip(e.targets, e.target_upper_bounds)}
                     })
        elif isinstance(e, Teg):

            if e.dvar in ctx.get('teg_sources', {}):
                ctx['teg_sources'].remove(e.dvar)
                target_dvar = ctx['teg_targets'].pop()
                placeholders = {
                    **{f'{svar.uid}_ub': upper for svar, (lower, upper) in ctx['source_bounds'].items()},
                    **{f'{svar.uid}_lb': lower for svar, (lower, upper) in ctx['source_bounds'].items()}
                }
                target_lower_bounds = resolve_placeholders(ctx['target_lower_bounds'][target_dvar], placeholders)
                target_upper_bounds = resolve_placeholders(ctx['target_upper_bounds'][target_dvar], placeholders)

                # Add new teg to list.
                ctx['new_tegs'] = [*ctx.get('new_tegs', []), (target_dvar, (target_lower_bounds, target_upper_bounds))]

                # Remove old teg.
                e = e.body

                if len(ctx['teg_sources']) == 0:
                    # Last teg replacement.

                    # Add let mappings here.
                    source_vars, source_exprs = zip(*list(ctx['let_mappings'].items()))
                    e = LetIn(source_vars, source_exprs, e)

                    # Add new tegs here.
                    for (new_dvar, (new_lb, new_ub)) in ctx['new_tegs']:
                        e = Teg(new_lb, new_ub, e, new_dvar)

                    # Add dependent mappings here.
                    for new_vars, new_exprs in ctx.get('dependent_mappings', []):
                        e = LetIn(new_vars, new_exprs, e)
                return e, ctx

        elif isinstance(e, LetIn):
            """
            if e.new_vars[0].name == '__norm__':
                print('FOUND NORM')
                print(e)
                print(ctx.get('teg_replace', {}))
                print(ctx.get('let_mappings', {}))
            """

            if len(ctx.get('teg_sources', {})) > 0:
                if (any([new_var in map_expr
                         for new_var in e.new_vars
                         for map_vars, map_exprs in ctx.get('dependent_mappings', [])
                         for map_expr in map_exprs]) or
                    any([new_var in map_expr
                         for new_var in e.new_vars
                         for map_var, map_expr in ctx.get('let_mappings', {}).items()])):

                    ctx['dependent_mappings'] = [*ctx.get('dependent_mappings', []), (e.new_vars, e.new_exprs)]
                    return e.expr, ctx

        return e, ctx

    def context_combine(contexts, ctx):
        ctxs = [context for context in contexts if 'teg_sources' in context]
        assert len(ctxs) <= 1, 'Found the same map in several branches'
        if len(ctxs) == 1:
            return {**ctxs[0], 'upper_tegvars': ctx['upper_tegvars']}
        elif len(ctxs) == 0:
            return ctx

    n_expr, context = base_pass(expr, {'upper_tegvars': set(), 'source_bounds': {}},
                                inner_fn, outer_fn, context_combine)
    return n_expr


def eliminate_bimaps(expr: ITeg):
    # find top_level bimap
    # check if bimap contains delta.
    # If yes: lift using split_instance() if bimap is not already linear in tree
    #         reduce using reparameterize()
    # If no: convert to let expression

    top_level_bimap = top_level_instance_of(expr, lambda a: isinstance(a, BiMap))
    if top_level_bimap is None:
        return expr

    top_level_delta_of_bimap = top_level_instance_of(top_level_bimap, lambda a: isinstance(a, Delta))
    if top_level_delta_of_bimap is None:
        let_expr = LetIn(top_level_bimap.targets, top_level_bimap.target_exprs, top_level_bimap.expr)
        return eliminate_bimaps(substitute_instance(expr, top_level_bimap, let_expr))
    else:
        linear_expr = split_instance(top_level_bimap, expr)

        old_tree = substitute_instance(expr, top_level_bimap, Const(0))
        new_tree = tree_copy(reparameterize(top_level_bimap, linear_expr))
        e = old_tree + new_tree

        return eliminate_bimaps(e)


def eliminate_deltas(expr: ITeg):
    # eliminate deltas through let expressions
    # remove the corresponding integral.
    # (error if corresponding integral does not exist)
    def inner_fn(e, ctx):
        if isinstance(e, Teg):
            return e, {'is_expr': ctx['search_expr'] is e,
                       'upper_tegvars': ctx['upper_tegvars'] | {e.dvar},
                       'search_expr': ctx['search_expr']}
        return e, {'is_expr': False,
                   'upper_tegvars': ctx['upper_tegvars'],
                   'search_expr': ctx['search_expr']}

    def outer_fn(e, ctx):
        if isinstance(e, Delta) and (ctx['search_expr'] is e):
            assert is_delta_normal(e), f'Delta {e} is not in normal form. Call normalize_delta() first'
            if e.expr not in ctx['upper_tegvars']:
                return Const(0), ctx
            else:
                return Const(1), {**ctx,
                                  'eliminate_tegs': {**ctx['eliminate_tegs'], e.expr: Const(0)}}

        elif isinstance(e, Teg):
            if e.dvar in ctx['eliminate_tegs']:
                value = ctx['eliminate_tegs'][e.dvar]
                bounds_check = (e.lower < value) & (e.upper > value)
                return (LetIn([e.dvar], [value], IfElse(bounds_check, e.body, Const(0))),
                        ctx)

        return e, ctx

    def context_combine(contexts, ctx):
        return {'lower_tegvars': reduce(lambda a, b: a | b, [ctx['lower_tegvars'] for ctx in contexts], set()),
                'upper_tegvars': ctx['upper_tegvars'],
                'eliminate_tegs': reduce(lambda a, b: {**a, **b}, [ctx['eliminate_tegs'] for ctx in contexts], {}),
                'search_expr': ctx['search_expr']}

    def eliminate_delta(delta, t_expr):
        return base_pass(t_expr,
                         {'upper_tegvars': set(), 'search_expr': delta},
                         inner_fn, outer_fn, context_combine)[0]

    top_level_delta = top_level_instance_of(expr, lambda a: isinstance(a, Delta))
    if top_level_delta is None:
        return expr
    else:
        linear_expr = split_instance(top_level_delta, expr)
        old_tree = substitute_instance(expr, top_level_delta, Const(0))
        new_tree = tree_copy(eliminate_delta(top_level_delta, linear_expr))
        return eliminate_deltas(old_tree + new_tree)
