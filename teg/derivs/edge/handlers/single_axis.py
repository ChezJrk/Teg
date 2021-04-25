from .affine import AffineHandler, extract_coefficients_from_affine, remove_constant_coeff
from teg import TegVar, Const
from teg.lang.extended import BiMap, Delta


def check_single_linear_var(expr, not_ctx=set()):
    """Checks that expr contains a single variable with coefficient 1. """
    affine_list = extract_coefficients_from_affine(expr, {(var.name, var.uid) for var in not_ctx})
    linear_list = remove_constant_coeff(affine_list)
    return (len(linear_list) == 1  # Single variable
            and Const(1) in linear_list.values())  # with a coefficient of 1


class ConstantAxisHandler(AffineHandler):
    """Handles expressons of the form Delta(x + c). """

    def can_rewrite(delta, not_ctx=set()):
        """Checks the delta expression is a TegVar and a constant. """
        try:
            return check_single_linear_var(delta.expr, not_ctx)
        except AssertionError:
            return False

    def rewrite(delta, not_ctx=set()):
        """Define a change of varibles so that Delta(x + c) becomes Delta(y). """
        affine_list = extract_coefficients_from_affine(delta.expr, {(var.name, var.uid) for var in not_ctx})
        constant = affine_list.get(('__const__', -1), Const(0))
        only_var = [(name, uid) for name, uid in affine_list.keys() if uid != -1]
        assert len(only_var) == 1, f'Only one tegvar can be included in the affine expression. {only_var}'

        var_name, var_uid = only_var[0]
        source_var = TegVar(name=var_name, uid=var_uid)
        target_var = TegVar(name=f'{var_name}_')
        return BiMap(expr=Delta(target_var),
                     sources=[source_var],
                     source_exprs=[target_var - constant],
                     targets=[target_var],
                     target_exprs=[source_var + constant],
                     inv_jacobian=Const(1),
                     target_lower_bounds=[source_var.lower_bound() + constant],
                     target_upper_bounds=[source_var.upper_bound() + constant])
