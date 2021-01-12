from typing import Dict, Set, List, Tuple
from functools import reduce, partial
from itertools import product
import operator

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Invert,
    IfElse,
    Teg,
    LetIn,
    TegVar,
    true,
    false,
)

from teg.math import (
    Sqrt,
    Sqr
)

from teg.lang.extended import (
    BiMap,
    Delta
)

from teg.passes.substitute import substitute

from .handler import DeltaHandler


class AffineHandler(DeltaHandler):

    def accept(delta, not_ctx=set()):
        try:
            affine_list = extract_coefficients_from_affine(delta.expr, not_ctx)
            assert all([is_expr_parametric(coeff, not_ctx) for coeff in affine_list.values()]), 'Coeffs not parametric'
            return True
        except AssertionError:
            return False
        # return check_affine(delta.expr, {(var.name, var.uid) for var in not_ctx})

    def rewrite(delta, not_ctx=set()):
        # Extract rotation.
        raw_affine_set = extract_coefficients_from_affine(delta.expr, {(var.name, var.uid) for var in not_ctx})

        # print('raw_affine_set: ', raw_affine_set)
        # Add a constant term if there isn't one.
        if ('__const__', -1) not in raw_affine_set:
            raw_affine_set[('__const__', -1)] = Const(0)

        # Extract source variables (in order).
        source_vars = [TegVar(name=name, uid=uid) for name, uid in var_list(affine_to_linear(raw_affine_set))]

        # Create rotated variables.
        target_vars = [TegVar(name=f'{var.name}_') for var in source_vars]

        # Identify innermost TegVar.
        # dvar_idx = source_vars.index(TegVar(name=expr.dvar.name, uid=expr.dvar.uid))
        dvar_idx = 0

        # Move this TegVar to position 0
        source_vars = [source_vars[dvar_idx]] + source_vars[:dvar_idx] + source_vars[dvar_idx + 1:]
        target_vars = [target_vars[dvar_idx]] + target_vars[:dvar_idx] + target_vars[dvar_idx + 1:]

        affine_set, flip_condition = negate_degenerate_coeffs(raw_affine_set, source_vars)
        linear_set = affine_to_linear(affine_set)
        normalized_set, normalization_var, normalization_expr = normalize_linear_set(linear_set)

        # Evaluate the discontinuity at x = t+
        # TODO: Rewrite so it works for a general reparameterization.
        dvar = target_vars[0]
        expr_for_dvar = -constant_coefficient(affine_set) * normalization_var

        source_exprs = [rotate_to_source(
                            normalized_set,
                            source_index=idx,
                            target_vars=target_vars,
                            source_vars=source_vars) for idx, source_var in enumerate(source_vars)]

        target_exprs = [rotate_to_target(
                            normalized_set,
                            target_index=idx,
                            target_vars=target_vars,
                            source_vars=source_vars) for idx, target_var in enumerate(target_vars)]

        lower_bounds, upper_bounds = zip(*[bounds_of(
                                        normalized_set,
                                        target_index=idx,
                                        target_vars=target_vars,
                                        source_vars=source_vars) for idx, target_var in enumerate(target_vars)])

        # print('AFFINE')
        # print(target_vars[0])
        # print(target_exprs[0])
        # print(dvar + expr_for_dvar)

        return LetIn([normalization_var], [normalization_expr],
                     BiMap(expr=Delta(dvar - expr_for_dvar),
                           sources=source_vars,
                           source_exprs=source_exprs,
                           targets=target_vars,
                           target_exprs=target_exprs,
                           inv_jacobian=normalization_var,
                           target_lower_bounds=lower_bounds,
                           target_upper_bounds=upper_bounds))


def transform_expr(expr, transforms: Dict[Tuple[str, int], ITeg]):
    for var_idname, transform in transforms.items():
        expr = substitute(expr, TegVar(name=var_idname[0], uid=var_idname[1]), transform)

    return expr


def combine_affine_sets(affine_lists: List[Dict[Tuple[str, int], ITeg]], op):
    # Combine affine sets. Assumes the list satisfies affine properties.
    combined_set = {}
    if op == operator.mul:
        # Cartesian product. Produce every variable combination.
        # TODO: Fix this asap..
        # print([ item for item in affine_lists[0].items() ])
        affine_products = product(*[affine_list.items() for affine_list in affine_lists])
        # print([ a for a in affine_products ])
        for affine_product in affine_products:
            combined_variable = [var_expr[0] for var_expr in affine_product]
            k = [var_expr[1] for var_expr in affine_product]
            combined_expr = reduce(operator.mul, k)

            # Reduce combined variables to primitive variables.
            primitive_variable = None
            is_sample = [sub_var != ('__const__', -1) for sub_var in combined_variable]
            if reduce(operator.or_, is_sample) is False:
                primitive_variable = ('__const__', -1)
            elif is_sample.count(True) == 1:
                primitive_variable = combined_variable[is_sample.index(True)]
            else:
                raise ValueError('Error when processing affine sets, \
                                  encountered non-affine combination.')

            combined_set[primitive_variable] = combined_expr

    elif op == operator.add:
        for affine_list in affine_lists:
            for variable, expr in affine_list.items():
                if variable in combined_set.keys():
                    combined_set[variable] = combined_set[variable] + expr
                else:
                    combined_set[variable] = expr

    else:
        raise ValueError('Operation not supported')

    return combined_set


def is_expr_parametric(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> bool:
    if isinstance(expr, TegVar):
        return False if (expr.name, expr.uid) in not_ctx else True
    elif isinstance(expr, (Var, Const)):
        return True
    else:
        return True

    return reduce(operator.and_, [is_expr_parametric(child, not_ctx) for child in expr.children])


def check_affine(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> bool:
    if isinstance(expr, Mul):
        cvals = [is_expr_parametric(child, not_ctx) for child in expr.children]
        return (cvals.count(False) == 1) and check_affine(expr.children[cvals.index(False)], not_ctx)
    elif isinstance(expr, Add):
        return reduce(operator.and_, [check_affine(child, not_ctx) for child in expr.children])
    elif isinstance(expr, TegVar) and (expr.name, expr.uid) in not_ctx:
        return True
    elif is_expr_parametric(expr, not_ctx):
        return True
    else:
        return False


def extract_coefficients_from_affine(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> Dict[Tuple[str, int], ITeg]:
    if isinstance(expr, Mul):
        children_coeffs = [extract_coefficients_from_affine(child, not_ctx) for child in expr.children]
        return combine_affine_sets(children_coeffs, op=operator.mul)
    elif isinstance(expr, Add):
        children_coeffs = [extract_coefficients_from_affine(child, not_ctx) for child in expr.children]
        return combine_affine_sets(children_coeffs, op=operator.add)
    elif isinstance(expr, TegVar) and (expr.name, expr.uid) in not_ctx:
        return {(expr.name, expr.uid): Const(1)}
    elif is_expr_parametric(expr, not_ctx):
        return {('__const__', -1): expr}
    else:
        return {('__const__', -1): Const(0)}


def constant_coefficient(affine_set: Dict[Tuple[str, int], ITeg]):
    if ('__const__', -1) in affine_set:
        return affine_set[('__const__', -1)]
    else:
        return Const(0)


def affine_to_linear(affine_set: Dict[Tuple[str, int], ITeg]):
    linear_set = dict(affine_set)
    if ('__const__', -1) in linear_set:
        linear_set.pop(('__const__', -1))
    return linear_set


def normalize_linear_set(linear_set: Dict[Tuple[str, int], ITeg]):
    normalization_var = Var('__norm__')
    normalization_expr = Invert(Sqrt(reduce(operator.add, [Sqr(expr) for var, expr in linear_set.items()])))

    normalized_set = {var: expr * normalization_var for var, expr in linear_set.items()}

    return normalized_set, normalization_var, normalization_expr


def negate_degenerate_coeffs(affine_set: Dict[Tuple[str, int], ITeg], source_vars: List[TegVar]):
    """
        If exprs[0] is negative, flip all coefficents so we don't run into
        degenerate conditions. Since coeffcients are not known in advance, this
        must be done using Teg instructions.
    """
    exprs = [affine_set[(s_var.name, s_var.uid)] for s_var in source_vars]

    flip_condition = exprs[0] < 0

    robust_affine_set = dict([(var, IfElse(flip_condition, -coeff, coeff)) for (var, coeff) in affine_set.items()])

    return robust_affine_set, flip_condition


def rotate_to_target(linear_set: Dict[Tuple[str, int], ITeg],
                     target_index: int,
                     target_vars: List[TegVar],
                     source_vars: List[TegVar]):

    num_vars = len(source_vars)
    exprs = [linear_set[(s_var.name, s_var.uid)] for s_var in source_vars]

    if target_index == 0:
        return sum(exprs[i] * source_vars[i] for i in range(num_vars))
    elif target_index < len(linear_set):
        i = target_index
        rotation_expr = sum(((Const(1) if i == j else Const(0))
                            - (exprs[i] * exprs[j]) / (1 + exprs[0])) * source_vars[j]
                            for j in range(1, num_vars))
        return -exprs[i] * source_vars[0] + rotation_expr
    else:
        raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')


def var_list(linear_set: Dict[Tuple[str, int], ITeg]):
    idnames = list(linear_set.keys())
    idnames.sort(key=lambda a: a[1])
    return idnames


def rotate_to_source(linear_set: Dict[Tuple[str, int], ITeg],
                     source_index: int,
                     target_vars: List[TegVar],
                     source_vars: List[TegVar]):

    num_vars = len(target_vars)
    exprs = [linear_set[(s_var.name, s_var.uid)] for s_var in source_vars]

    if source_index == 0:
        return sum((Const(1) if i == 0 else Const(-1)) * exprs[i] * target_vars[i] for i in range(num_vars))
    elif source_index < len(linear_set):
        i = source_index
        # TODO: Potential change here.. (not considering the top row)
        inverse_rotation = sum(((Const(1) if i == j else Const(0))
                               - (exprs[i] * exprs[j]) / (1 + exprs[0])) * target_vars[j]
                               for j in range(1, num_vars))
        return inverse_rotation + exprs[i] * target_vars[0]
    else:
        raise ValueError(f'Requested source coordinate index: {source_index} is invalid.')


def bounds_of(linear_set: Dict[Tuple[str, int], ITeg],
              target_index: int,
              target_vars: List[TegVar],
              source_vars: List[TegVar]):
    # Return bounds using placeholders.
    # idnames = affine_set.keys()
    # sorted_idnames = sort(idnames, key=lambda a:a[1])

    num_vars = len(source_vars)
    exprs = [linear_set[(s_var.name, s_var.uid)] for s_var in source_vars]

    if target_index == 0:
        # raise ValueError('Bounds for target variable 0 are undefined.')
        return sum(exprs[i] * IfElse(exprs[i] > 0, source_vars[i].lower_bound(), source_vars[i].upper_bound())
                   for i in range(num_vars)),\
               sum(exprs[i] * IfElse(exprs[i] > 0, source_vars[i].upper_bound(), source_vars[i].lower_bound())
                   for i in range(num_vars))
    elif target_index < len(linear_set):
        def coeff(u, v):
            if v == 0:
                return -exprs[u]
            else:
                return ((Const(1) if u == v else Const(0)) -
                        (exprs[u] * exprs[v]) / (Const(1) + exprs[0]))

        i = target_index
        lower_bound = upper_bound = Const(0)
        for j in range(num_vars):
            placeholder_lb = source_vars[j].lower_bound()
            placeholder_ub = source_vars[j].upper_bound()
            lower_bound += coeff(i, j) * IfElse(coeff(i, j) > 0, placeholder_lb, placeholder_ub)
            upper_bound += coeff(i, j) * IfElse(coeff(i, j) > 0, placeholder_ub, placeholder_lb)
        return lower_bound, upper_bound
    else:
        raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')