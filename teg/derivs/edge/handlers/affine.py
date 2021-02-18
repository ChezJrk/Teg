from typing import Dict, Set, List, Tuple, Optional, Union
from functools import reduce
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
    LetIn,
    TegVar,
)

from teg.math import (
    Sqrt,
    Sqr
)

from teg.lang.extended import (
    BiMap,
    Delta
)

from .handler import DeltaHandler


class AffineHandler(DeltaHandler):
    """Accounts for affine discontinuities"""

    def can_rewrite(delta: Delta, not_ctx: Optional[Set] = None) -> bool:
        """Check if a given expression is affine. """
        not_ctx = set() if not_ctx is None else not_ctx
        try:
            vars_to_coeffs = extract_coefficients_from_affine(delta.expr, {(var.name, var.uid) for var in not_ctx})
            error_message = 'Coefficients should not contain variables of integration'
            assert all([is_expr_parametric(coeff, not_ctx) for coeff in vars_to_coeffs.values()]), error_message
            return True
        except AssertionError:
            return False

    def rewrite(delta: Delta, not_ctx: Optional[Set] = set()) -> ITeg:
        """Rotates an affine discontinuity so that it's axis-aligned (e.g. ax + by + c -> z + d). """

        not_ctx = set() if not_ctx is None else not_ctx

        # Canonicalize affine expression into a map {var: coeff}
        raw_affine_set = extract_coefficients_from_affine(delta.expr, not_ctx)

        # Introduce a constant term if there isn't one
        if ('__const__', -1) not in raw_affine_set:
            raw_affine_set[('__const__', -1)] = Const(0)

        # Extract source variables (in order)
        source_vars = [TegVar(name=name, uid=uid) for name, uid in var_list(remove_constant_coeff(raw_affine_set))]

        # Create rotated (target) variables
        target_vars = [TegVar(name=f'{var.name}_') for var in source_vars]

        # TODO: Currently, do not handle degeneracy at -1
        affine_set, flip_condition = negate_degenerate_coeffs(raw_affine_set, source_vars)
        linear_set = remove_constant_coeff(affine_set)
        normalized_set, normalization_var, normalization_expr = normalize_linear(linear_set)

        dvar = target_vars[0]
        expr_for_dvar = -constant_coeff(affine_set) * normalization_var

        source_exprs = rotate_to_source(normalized_set, target_vars, source_vars)
        target_exprs = rotate_to_target(normalized_set, source_vars)
        lower_bounds, upper_bounds = bounds_of(normalized_set, source_vars)

        return LetIn([normalization_var], [normalization_expr],
                     BiMap(expr=Delta(dvar - expr_for_dvar),
                           sources=source_vars,
                           source_exprs=source_exprs,
                           targets=target_vars,
                           target_exprs=target_exprs,
                           inv_jacobian=normalization_var,
                           target_lower_bounds=lower_bounds,
                           target_upper_bounds=upper_bounds))


def combine_affine_sets(affine_lists: List[Dict[Tuple[str, int], ITeg]], op) -> List[Dict[Tuple[str, int], ITeg]]:
    """Computes the canonical version of a list of affine expressions using the operator 'op'. """

    combined_set = {}
    if op == operator.mul:
        # NOTE: this could be made faster by identifying one expression as a constant expression
        # and doing a linear product instead of the quadratic Cartesian product

        # Distribute out expression with a Cartesian product
        affine_products = product(*[affine_list.items() for affine_list in affine_lists])
        for affine_product in affine_products:
            combined_variable = [var_expr[0] for var_expr in affine_product]
            k = [var_expr[1] for var_expr in affine_product]
            combined_expr = reduce(operator.mul, k)

            # Reduce combined variables to primitive variables
            primitive_variable = None
            is_sample = [sub_var != ('__const__', -1) for sub_var in combined_variable]
            if reduce(operator.or_, is_sample) is False:
                primitive_variable = ('__const__', -1)
            elif is_sample.count(True) == 1:
                primitive_variable = combined_variable[is_sample.index(True)]
            else:
                raise AssertionError('Error when processing affine sets, encountered non-affine product.')

            combined_set[primitive_variable] = combined_expr

    elif op == operator.add:
        for affine_list in affine_lists:
            for variable, expr in affine_list.items():
                if variable in combined_set.keys():
                    combined_set[variable] = combined_set[variable] + expr
                else:
                    combined_set[variable] = expr

    else:
        raise AssertionError('Operation not supported.')

    return combined_set


def is_expr_parametric(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> bool:
    """Checks whether an expression contains a variable of integration. """
    if isinstance(expr, TegVar):
        return False if (expr.name, expr.uid) in not_ctx else True
    elif isinstance(expr, (Var, Const)):
        return True
    else:
        return reduce(operator.and_, [is_expr_parametric(child, not_ctx) for child in expr.children])


def extract_coefficients_from_affine(expr: ITeg, not_ctx: Set[Union[Var, Tuple]]) -> Dict[Tuple[str, int], ITeg]:
    """Canonicalizes an affine expression to a mapping from variables to coefficients with a constant term. """
    if isinstance(expr, Mul):
        children_coeffs = [extract_coefficients_from_affine(child, not_ctx) for child in expr.children]
        return combine_affine_sets(children_coeffs, op=operator.mul)
    elif isinstance(expr, Add):
        children_coeffs = [extract_coefficients_from_affine(child, not_ctx) for child in expr.children]
        return combine_affine_sets(children_coeffs, op=operator.add)
    elif isinstance(expr, TegVar) and expr in not_ctx:
        return {(expr.name, expr.uid): Const(1)}
    elif is_expr_parametric(expr, not_ctx):
        return {('__const__', -1): expr}
    else:
        return {('__const__', -1): Const(0)}


def constant_coeff(affine: Dict[Tuple[str, int], ITeg]):
    """Extract the constant coefficient if it exists, otherwise, return 0. """
    return affine[('__const__', -1)] if ('__const__', -1) in affine else Const(0)


def remove_constant_coeff(affine: Dict[Tuple[str, int], ITeg]) -> Dict[Tuple[str, int], ITeg]:
    """Remove the constant coefficient from an affine set. """
    linear_set = dict(affine)
    if ('__const__', -1) in linear_set:
        linear_set.pop(('__const__', -1))
    return linear_set


def normalize_linear(linear: Dict[Tuple[str, int], ITeg]):
    """Normalizes the coefficients vector of a linear expression. """
    normalization_var = Var('__norm__')
    normalization_expr = Invert(Sqrt(reduce(operator.add, [Sqr(expr) for var, expr in linear.items()])))
    normalized_set = {var: expr * normalization_var for var, expr in linear.items()}

    return normalized_set, normalization_var, normalization_expr


def negate_degenerate_coeffs(affine: Dict[Tuple[str, int], ITeg], source_vars: List[TegVar]):
    """Flips all the coefficients if expr[0] < 0 to avoid degeneracies"""
    exprs = [affine[(s_var.name, s_var.uid)] for s_var in source_vars]

    flip_condition = exprs[0] < 0

    robust_affine_set = dict([(var, IfElse(flip_condition, -coeff, coeff)) for (var, coeff) in affine.items()])

    return robust_affine_set, flip_condition


def rotate_to_target(linear: Dict[Tuple[str, int], ITeg], source_vars: List[TegVar]) -> List[ITeg]:
    """Generates the set of expressions for the rotated target variables.

    See Appendix A for details.
    """
    rotation = []
    num_vars = len(source_vars)
    exprs = [linear[(s_var.name, s_var.uid)] for s_var in source_vars]
    for target_index in range(num_vars):
        if target_index == 0:
            rotation.append(sum(exprs[i] * source_vars[i] for i in range(num_vars)))
        elif target_index < len(linear):
            i = target_index
            rotation_expr = sum(((Const(1) if i == j else Const(0))
                                - (exprs[i] * exprs[j]) / (1 + exprs[0])) * source_vars[j]
                                for j in range(1, num_vars))
            rotation.append(-exprs[i] * source_vars[0] + rotation_expr)
        else:
            raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')

    return rotation


def var_list(linear: Dict[Tuple[str, int], ITeg]):
    """Extracts the sorted variable names from a linear expression. """
    idnames = list(linear.keys())
    idnames.sort(key=lambda a: a[1])
    return idnames


def rotate_to_source(linear: Dict[Tuple[str, int], ITeg],
                     target_vars: List[TegVar],
                     source_vars: List[TegVar]) -> List[ITeg]:
    """Generates the set of expressions for the source variables in terms of the rotated targets.

    See Appendix A for details.
    """
    rotation = []
    num_vars = len(target_vars)
    exprs = [linear[(s_var.name, s_var.uid)] for s_var in source_vars]
    for source_index in range(num_vars):
        if source_index == 0:
            rotation.append(sum((Const(1) if i == 0 else Const(-1)) * exprs[i] * target_vars[i] for i in range(num_vars)))
        elif source_index < len(linear):
            i = source_index
            inverse_rotation = sum(((Const(1) if i == j else Const(0))
                                   - (exprs[i] * exprs[j]) / (1 + exprs[0])) * target_vars[j]
                                   for j in range(1, num_vars))
            rotation.append(inverse_rotation + exprs[i] * target_vars[0])
        else:
            raise ValueError(f'Requested source coordinate index: {source_index} is invalid.')

    return rotation


def bounds_of(linear: Dict[Tuple[str, int], ITeg], source_vars: List[TegVar]) -> List[ITeg]:
    """Generates the bounds of integration after rotation (i.e., it's the bounds transfer function). """
    lower_bounds, upper_bounds = [], []
    num_vars = len(source_vars)
    exprs = [linear[(s_var.name, s_var.uid)] for s_var in source_vars]
    for target_index in range(num_vars):
        if target_index == 0:
            lower = sum(exprs[i] * IfElse(exprs[i] > 0, source_vars[i].lower_bound(), source_vars[i].upper_bound())
                        for i in range(num_vars))
            upper = sum(exprs[i] * IfElse(exprs[i] > 0, source_vars[i].upper_bound(), source_vars[i].lower_bound())
                        for i in range(num_vars))
        elif target_index < len(linear):
            def coeff(u, v):
                if v == 0:
                    return -exprs[u]
                else:
                    return ((Const(1) if u == v else Const(0)) - (exprs[u] * exprs[v]) / (Const(1) + exprs[0]))

            i = target_index
            lower = upper = Const(0)
            for j in range(num_vars):
                placeholder_lb = source_vars[j].lower_bound()
                placeholder_ub = source_vars[j].upper_bound()
                lower += coeff(i, j) * IfElse(coeff(i, j) > 0, placeholder_lb, placeholder_ub)
                upper += coeff(i, j) * IfElse(coeff(i, j) > 0, placeholder_ub, placeholder_lb)
        else:
            raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return lower_bounds, upper_bounds
