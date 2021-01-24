from typing import Dict, Set, List, Tuple
from functools import reduce, partial, cmp_to_key
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
from teg.maps.transform import scale, translate

from .handler import DeltaHandler

CONST_VAR = Var('__const__', uid=-1)
MIN_T = 10e-5
MAX_T = 10e5


def combine_variables(var_lists: List[Tuple[Tuple[Var, int]]]):
    multiplicities = {}
    for var_list in var_lists:
        for var, multiplicity in var_list:
            multiplicities.setdefault(var, 0)
            multiplicities[var] += multiplicity
    new_var = ((var, multiplicity) for var, multiplicity in multiplicities.items()
               if var is not CONST_VAR and multiplicity != 0)
    new_var = tuple(sorted(new_var, key=lambda a: a[0].uid))
    return new_var if new_var else ((CONST_VAR, 1),)


def get_poly_term(expr_list, multiplicities):
    new_var = ((var, multiplicity) for var, multiplicity in multiplicities.items()
               if var is not CONST_VAR and multiplicity != 0)
    new_var = tuple(sorted(new_var, key=lambda a: a[0].uid))
    new_var = new_var if new_var else ((CONST_VAR, 1),)
    return expr_list.get(new_var, Const(0))


def combine_poly_sets(poly_lists: List[Dict[Tuple[Tuple[Var, int]], ITeg]], op):
    # Combine polynomial sets. Assumes the list satisfies affine properties.
    combined_set = {}
    if op == operator.mul:
        # Cartesian product. Produce every variable combination.
        # print([ item for item in affine_lists[0].items() ])
        poly_products = product(*[poly_list.items() for poly_list in poly_lists])
        # print([ a for a in affine_products ])

        for poly_product in poly_products:
            combined_variable = [var_expr[0] for var_expr in poly_product]
            k = [var_expr[1] for var_expr in poly_product]
            combined_expr = reduce(operator.mul, k)

            # Reduce combined variables to primitive variables.
            primitive_variable = combine_variables(combined_variable)
            combined_set[primitive_variable] = combined_expr

    elif op == operator.add:
        for poly_list in poly_lists:
            for variable, expr in poly_list.items():
                combined_set[variable] = combined_set.get(variable, Const(0)) + expr

    else:
        raise ValueError('Operation not supported')

    return combined_set


def extract_coefficients_from_polynomial(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> Dict[Set[Tuple[Var, int]], ITeg]:
    if isinstance(expr, Mul):
        children_coeffs = [extract_coefficients_from_polynomial(child, not_ctx) for child in expr.children]
        return combine_poly_sets(children_coeffs, op=operator.mul)
    elif isinstance(expr, Add):
        children_coeffs = [extract_coefficients_from_polynomial(child, not_ctx) for child in expr.children]
        return combine_poly_sets(children_coeffs, op=operator.add)
    elif isinstance(expr, TegVar) and (expr.name, expr.uid) in not_ctx:
        return {((expr, 1),): Const(1)}
    elif is_expr_parametric(expr, not_ctx):
        return {((CONST_VAR, 1),): expr}
    else:
        return {((CONST_VAR, 1),): Const(0)}


def teg_max(a, b):
    return IfElse(a > b, a, b)


def teg_min(a, b):
    return IfElse(a > b, b, a)


def teg_abs(a):
    return IfElse(a > 0, a, -a)


def teg_cases(exprs, conditions):
    assert len(exprs) - len(conditions) == 1, 'Need one additional expression for the default step'

    # Build ladder in reverse order.s
    exprs = exprs[::-1]
    conditions = conditions[::-1]

    if_else_ladder = exprs[0]
    for expr, condition in zip(exprs[1:], conditions):
        if_else_ladder = IfElse(condition, expr, if_else_ladder)

    return if_else_ladder


class BilinearHandler(DeltaHandler):

    def accept(delta, not_ctx=set()):
        try:
            poly_list = extract_coefficients_from_polynomial(delta.expr, {(var.name, var.uid) for var in not_ctx})
            # print(poly_list.keys())
            assert all([is_expr_parametric(coeff, not_ctx) for coeff in poly_list.values()]),\
                   'Coeffs not parametric'
            assert all([multiplicity < 2 for term in poly_list for var, multiplicity in term]),\
                   'Contains a squared element'
            unique_vars = set()
            for term in poly_list:
                for var, multiplicity in term:
                    if var is not CONST_VAR:
                        unique_vars.add(var)

            assert len(unique_vars) == 2, 'Needs exactly two variables of integration to be bilinear'
            assert any([len({var for var, multiplicity in term}) == 2 for term in poly_list]),\
                   'Must contain at least one cross term "xy"'

            return True
        except AssertionError:
            return False

    def rewrite(delta, not_ctx=set()):
        # Extract polynomial coefficients.
        poly_set = extract_coefficients_from_polynomial(delta.expr, {(var.name, var.uid) for var in not_ctx})

        unique_vars = []
        for term in poly_set:
            for var, multiplicity in term:
                if var is not CONST_VAR:
                    unique_vars.append(var)

        x = unique_vars[0]
        y = unique_vars[1]

        c_xy = get_poly_term(poly_set, {x: 1, y: 1})  # poly_set.get({(x, 1), (y, 1)}, Const(0))
        c_x = get_poly_term(poly_set, {x: 1})
        c_y = get_poly_term(poly_set, {y: 1})
        c_1 = get_poly_term(poly_set, {})

        c_xy_var = Var(f'c_{x.name}_{y.name}')
        c_x_var = Var(f'c_{x.name}')
        c_y_var = Var(f'c_{y.name}')
        c_1_var = Var('c_1')

        coeff_vars = [c_xy_var, c_x_var, c_y_var, c_1_var]
        coeff_exprs = [teg_abs(c_xy),
                       IfElse(c_xy > 0, c_x, -c_x),
                       IfElse(c_xy > 0, c_y, -c_y),
                       IfElse(c_xy > 0, c_1, -c_1)]

        # print(f'Coeffs: {c_xy} * xy + {c_x} * x + {c_y} * y + {c_1}')
        # Sqrt c_xy
        sqrt_c_xy = Sqrt(c_xy_var)
        sqrt_c_xy_var = Var(f'{x.name}_{y.name}_sqrt')
        # c_xy_var = Var(f'{x.name}_{y.name}')

        needs_transforms = (c_x != Const(0) or c_y != Const(0))

        if needs_transforms:
            scale_map = partial(scale,
                                scale=[sqrt_c_xy_var, sqrt_c_xy_var])

            translate_map = partial(translate,
                                    translate=[c_y_var/sqrt_c_xy_var, c_x_var/sqrt_c_xy_var])

            scaler, (x_s, y_s) = scale_map([x, y])
            translater, (x_st, y_st) = translate_map([x_s, y_s])

            sqr_constant = (c_x_var * c_y_var) / (c_xy_var) - c_1_var
            # is_valid = ((c_x * c_y) / (c_xy_var)) > c_1
            scale_jacobian = Const(1)
        else:
            x_st, y_st = x, y
            sqr_constant = -c_1_var / c_xy_var
            # is_valid = c_1 < 0
            scale_jacobian = c_xy_var

        # If threshold is negative, the hyperbola is in the second and fourth quadrants.
        # Inverting either one of x or y automatically handles this.
        conditional_inverter, (x_st,) = scale([x_st], scale=[IfElse(sqr_constant > 0, 1, -1)])
        adjusted_sqr_constant = teg_abs(sqr_constant)  # IfElse(sqr_constant > 0, sqr_constant, -sqr_constant)
        constant = Sqrt(adjusted_sqr_constant)

        # Hyperbolic transform
        hyp_a, hyp_t = TegVar('hyp_a'), TegVar('hyp_t')

        # Build bounds transfer expressions.
        pos_a_lb = teg_cases([Sqrt(x_st.lb() * y_st.lb()), Const(0)],
                             [(x_st.lb() > 0) & (y_st.lb() > 0)])

        pos_a_ub = teg_cases([Sqrt(x_st.ub() * y_st.ub()), Const(0)],
                             [(x_st.ub() > 0) & (y_st.ub() > 0)])

        neg_a_lb = teg_cases([-Sqrt(x_st.lb() * y_st.lb()), Const(0)],
                             [(x_st.lb() < 0) & (y_st.lb() < 0)])
        neg_a_ub = teg_cases([-Sqrt(x_st.ub() * y_st.ub()), Const(0)],
                             [(x_st.ub() < 0) & (y_st.ub() < 0)])

        pos_t_lb = teg_max(
                    teg_cases([teg_max(x_st.lb()/hyp_a, hyp_a/y_st.ub()), hyp_a/y_st.ub(), MIN_T],
                              [(y_st.ub() > 0) & (x_st.lb() > 0), y_st.ub() > 0]), MIN_T
                    )

        pos_t_ub = teg_min(
                    teg_cases([teg_min(x_st.ub()/hyp_a, hyp_a/y_st.lb()), x_st.ub()/hyp_a, MAX_T],
                              [(x_st.ub() > 0) & (y_st.lb() > 0), x_st.ub() > 0]), MAX_T
                    )

        neg_t_lb = teg_max(
                    teg_cases([teg_max(x_st.ub()/hyp_a, hyp_a/y_st.lb()), hyp_a/y_st.lb(), MIN_T],
                              [(y_st.lb() < 0) & (x_st.ub() < 0), y_st.lb() < 0]), MIN_T
                    )
        neg_t_ub = teg_min(
                    teg_cases([teg_min(x_st.lb()/hyp_a, hyp_a/y_st.ub()), x_st.lb()/hyp_a, MAX_T],
                              [(x_st.lb() < 0) & (y_st.ub() < 0), x_st.lb() < 0]), MAX_T
                    )

        pos_curve = BiMap(Delta(hyp_a - constant),
                          sources=[x_st, y_st],
                          source_exprs=[hyp_a * hyp_t, hyp_a / hyp_t],
                          targets=[hyp_a, hyp_t],
                          target_exprs=[Sqrt(x_st * y_st), Sqrt(x_st / y_st)],
                          inv_jacobian=(hyp_a / hyp_t) * (1/(constant * scale_jacobian)),
                          target_lower_bounds=[pos_a_lb, pos_t_lb],
                          target_upper_bounds=[pos_a_ub, pos_t_ub])

        neg_curve = BiMap(Delta(hyp_a + constant),
                          sources=[x_st, y_st],
                          source_exprs=[hyp_a * hyp_t, hyp_a / hyp_t],
                          targets=[hyp_a, hyp_t],
                          target_exprs=[-Sqrt(x_st * y_st), Sqrt(x_st / y_st)],
                          inv_jacobian=(-1 * hyp_a / hyp_t) * (1/(constant * scale_jacobian)),
                          target_lower_bounds=[neg_a_lb, neg_t_lb],
                          target_upper_bounds=[neg_a_ub, neg_t_ub])

        if needs_transforms:
            return LetIn(coeff_vars, coeff_exprs, LetIn([sqrt_c_xy_var], [sqrt_c_xy],
                         scaler(translater(conditional_inverter(pos_curve + neg_curve)))))
        else:
            return LetIn(coeff_vars, coeff_exprs,
                         conditional_inverter(pos_curve + neg_curve))


def is_expr_parametric(expr: ITeg, not_ctx: Set[Tuple[str, int]]) -> bool:
    if isinstance(expr, TegVar):
        return False if (expr.name, expr.uid) in not_ctx else True
    elif isinstance(expr, (Var, Const)):
        return True
    else:
        return True

    return reduce(operator.and_, [is_expr_parametric(child, not_ctx) for child in expr.children])
