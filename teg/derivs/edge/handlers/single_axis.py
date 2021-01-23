from .affine import AffineHandler, extract_coefficients_from_affine, affine_to_linear
from teg.passes.base import base_pass


from teg import (
    TegVar,
    Add,
    Mul,
    Const
)

from teg.lang.extended import (
    BiMap,
    Delta
)


def check_single_linear_var(expr, not_ctx=set()):
    """
    def inner_fn(expr, context):
        if isinstance(expr, TegVar):
            return expr, {'var': expr.var, 'is_slv': True}
        else:
            return expr, {'is_slv': True}

    def outer_fn(expr, context):
        if isinstance(expr, Add):
            return expr, {'var': context['vars'][0],
                          'is_slv': all(context['is_slvlist']) and len(set(context)) == 1}
        elif isinstance(expr, Mul):
            return expr, {'var': context['vars'][0],
                          'is_slv': all(context['is_slvlist']) and len(context) == 1}
        else:
            return expr, {'is_slv': False}

    def combine_contexts(contexts):
        varlist = [context['var'] for context in contexts if 'var' in context]
        is_slvlist = [context['is_slv'] for context in contexts if 'is_slv' in context]
        return {'vars': varlist, 'is_slvlist': is_slvlist}

    expr, context = base_pass(expr, {}, inner_fn, outer_fn, combine_contexts)
    return context['is_slv']
    """
    # print(not_ctx)
    affine_list = extract_coefficients_from_affine(expr, {(var.name, var.uid) for var in not_ctx})
    # print(affine_list)
    linear_list = affine_to_linear(affine_list)
    return (len(linear_list) == 1  # Single variable
            and Const(1) in linear_list.values())  # with a coefficient of 1


class ConstantAxisHandler(AffineHandler):
    """
        Handles expressons of the form (Delta[x + c > 0])
    """

    def accept(delta, not_ctx=set()):
        # Check if there is only one TegVar and one constant
        try:
            return check_single_linear_var(delta.expr, not_ctx)
        except AssertionError:
            return False

    def rewrite(delta, not_ctx=set()):
        affine_list = extract_coefficients_from_affine(delta.expr, {(var.name, var.uid) for var in not_ctx})
        constant = affine_list.get(('__const__', -1), Const(0))
        only_var = [(name, uid) for name, uid in affine_list.keys() if uid != -1]
        assert len(only_var) == 1, f'Only one tegvar can be included in the affine expression. {only_var}'

        var_name, var_uid = only_var[0]
        source_var = TegVar(name=var_name, uid=var_uid)
        target_var = TegVar(name=f'{var_name}_')
        return BiMap(expr=Delta(target_var),
                     targets=[target_var],
                     target_exprs=[source_var + constant],
                     sources=[source_var],
                     source_exprs=[target_var - constant],
                     inv_jacobian=Const(1),
                     target_upper_bounds=[source_var.upper_bound() + constant],
                     target_lower_bounds=[source_var.lower_bound() + constant])
