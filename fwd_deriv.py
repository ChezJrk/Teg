from typing import Dict, Set, List, Tuple, Iterable
from functools import reduce
from itertools import product
import operator

from integrable_program import (
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
    Ctx,
    ITegBool,
    Bool,
    And,
    Or,
    true,
    false,
)
from substitute import substitute

"""
def extract_constants_from_affine(expr: ITeg) -> List[ITeg]:

    if isinstance(expr, Const):
        return [expr]

    elif isinstance(expr, Var):
        return []

    elif isinstance(expr, (Add, Mul)):
        return extract_constants_from_affine(expr.children[0]) + extract_constants_from_affine(expr.children[1])

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')


def extract_variables_from_affine(expr: ITeg) -> Dict[Tuple[str, int], ITeg]:

    if isinstance(expr, Const):
        return {}

    elif isinstance(expr, Var):
        return {(expr.name, expr.uid): expr}

    elif isinstance(expr, (Add, Mul)):
        return {**extract_variables_from_affine(expr.children[0]), **extract_variables_from_affine(expr.children[1])}

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')
"""

def combine_affine_sets(affine_list: Dict[Tuple[str, int], ITeg], op):
    # Combine affine sets. Assumes the list satisfies affine properties.
    combined_set = {}
    if op == operator.mul:
        affine_products = product(affine_list.items())
        for affine_product in affine_products:
            combined_variable = reduce(affine_product.keys(), operator.sum)
            combined_expr = reduce(affine_product.keys(), operator.mul)

            # Reduce combined variables to primitive variables.
            primitive_variable = None
            is_sample = [ for sub_var in combined_variable if sub_var != -1 ]
            if reduce(is_sample, operator.or) == False:
                primitive_variable = -1
            elif is_sample.count(True) == 1:
                primitive_variable = combined_variable[is_sample.index(True)]
            else:
                raise ValueError(f'Error when processing affine sets, encountered non-affine combination.')

            combined_set[primitive_variable] = combined_expr

    elif op == operator.add:
        for affine_set in affine_list:
            for variable, expr in affine_set:
                if variable in combined_set.keys():
                    combined_set[variable] = combined_set[variable] + expr
                else:
                    combined_set[variable] = expr

    else:
        raise ValueError(f'Operation not supported')

    return combined_set


def is_expr_parametric(expr: ITeg) -> bool:
    if isinstance(expr, Sample):
        return False
    elif isinstance(expr, (Parameter, Const)):
        return True
    else:
        return True

    return reduce([ is_expr_parametric(child) for child in expr.children ], operator.and)


def check_affine_tree(expr: ITeg) -> bool:
    if isinstance(expr, Mul):
        cvals = [ is_expr_parameteric(child) for child in expr.children ]
        return (cvals.count(False) == 1) and check_affine_tree(expr.children[cvals.index(False)])
    elif isinstance(expr, Add): 
        return reduce([ check_affine_tree(child) for child in expr.children ], operator.and)
    elif isinstance(expr, TegVar):
        return True
    elif is_expr_parameteric(expr):
        return True
    else:
        return False


def extract_coefficients_from_affine_tree(expr: ITeg) -> List[ITeg]:
    if isinstance(expr, Mul):
        children_coeffs = [ extract_coefficients_from_affine_tree(child) for child in expr.children ]
        return combine_affine_sets(children_coeffs, op=operator.mul)
    elif isinstance(expr, Add): 
        children_coeffs = [ extract_coefficients_from_affine_tree(child) for child in expr.children ]
        return combine_affine_sets(children_coeffs, op=operator.add)
    elif isinstance(expr, TegVar):
        return {(expr.name, expr.uid): Const(1)}
    elif is_expr_parameteric(expr):
        return {('__const__', -1): expr}
    else:
        return {('__const__', -1): Const(0)}

def affine_to_linear(affine_set: Dict[Tuple[str, int], ITeg]):
    linear_set = dict(affine_set)
    linear_set.pop(('__const', -1))

def rotate_to_target(linear_set: Dict[Tuple[str, int], ITeg], target_index: int):
    idnames = linear_set.keys()
    sorted_idnames = var_list(linear_set)

    num_vars = len(sorted_idnames)
    exprs = [linear_set[idname] for idname in sorted_idnames]

    # source_vars[i] = TegVar(name=f'a_{i}', id=sorted_ids[i])

    if target_index == 0:
        name, id = sorted_idnames[i]
        return reduce([exprs[i] * TegVar(name=name, id=id) for i in range(num_vars)], operator.add)
    elif target_index < len(linear_set):
        i = target_index
        return reduce([ (1 if i == j else 0) - 
                (exprs[i] * exprs[j]) / (1 + exprs[0]) \
                * TegVar(name=sorted_idnames[j][0], id=sorted_idnames[j][1]) \
                for j in range(num_vars)], operator.add)
    else:
        raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')

def var_list(linear_set):
    idnames = linear_set.keys()
    sorted_idnames = sort(idnames, key=lambda a:a[1])
    return sorted_idnames

def rotate_to_source(linear_set: Dict[Tuple[str, int], ITeg], source_index: int, target_vars: List[TegVar]):
    idnames = linear_set.keys()
    sorted_idnames = var_list(linear_set)

    num_vars = len(sorted_idnames)
    exprs = [linear_set[idname] for idname in sorted_idnames]

    if source_index == 0:
        return reduce([(-1 if sid == 0 else 1) * exprs[i] * target_vars[i] for i in range(num_vars)], operator.add)
    elif source_index < len(linear_set):
        i = source_index
        return reduce([ (1 if i == j else 0) - 
                (exprs[i] * exprs[j]) / (1 + exprs[0]) \
                * target_vars[j] \
                for j in range(num_vars)], operator.add)
    else:
        raise ValueError(f'Requested source coordinate index: {source_index} is invalid.')

def bounds_of(affine_set: Dict[Tuple[str, int], ITeg], target_index: int):
    # Return bounds using placeholders.
    idnames = affine_set.keys()
    sorted_idnames = sort(idnames, key=lambda a:a[1])
    num_vars = len(sorted_idnames)
    exprs = [affine_set[idname] for idname in sorted_idnames]

    if target_index == 0:
        raise ValueError(f'Requested target coordinate index: {target_index} is invalid.')
    elif target_index < len(affine_set):
        i = target_index
        coeff = (1 if i == j else 0) - 
                (exprs[i] * exprs[j]) / (1 + exprs[0])
        return  reduce([ coeff \
                    * IfElse( coeff > 0, 
                            Placeholder(signature=f"{sorted_idnames[j][0]}_lb"), 
                            Placeholder(signature=f"{sorted_idnames[j][0]}_ub"), 
                        ) \
                for j in range(num_vars)], operator.add),
                reduce([ coeff \
                    * IfElse( coeff > 0, 
                            Placeholder(signature=f"{sorted_idnames[j][0]}_ub"), 
                            Placeholder(signature=f"{sorted_idnames[j][0]}_lb"), 
                        ) \
                for j in range(num_vars)], operator.add)
    else:
        raise ValueError(f'Requested target coordinate index: {target_index} is out of bounds.')


def extract_variables_from_affine(expr: ITeg) -> Dict[Tuple[str, int], ITeg]:

    if isinstance(expr, Const):
        return {}

    elif isinstance(expr, Var):
        return {(expr.name, expr.uid): expr}

    elif isinstance(expr, (Add, Mul)):
        return {**extract_variables_from_affine(expr.children[0]), **extract_variables_from_affine(expr.children[1])}

    else:
        raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')

def solve_for_dvar(expr: ITeg, var: ITeg):

    def flatten_to_nary_add(expr: ITeg) -> List[ITeg]:

        if isinstance(expr, Var):
            return [expr]

        elif isinstance(expr, Mul):
            if expr.children[0] == Const(-1):
                return [-e for e in flatten_to_nary_add(expr.children[1])]
            else:
                return [expr]

        elif isinstance(expr, Add):
            return [e for ex in expr.children for e in flatten_to_nary_add(ex)]

        else:
            raise ValueError(f'The expression of type "{type(expr)}" results in a computation that is not affine.')

    def prod(l: List[ITeg]) -> ITeg:
        # NOTE: Butchering constants...
        return reduce(lambda x, y: x * y, [c.value for c in l], 1)

    # Combine all common terms in the constant vector pairs
    d, const = {}, 0
    for e in flatten_to_nary_add(expr):
        vs = extract_variables_from_affine(e).values()
        cs = extract_constants_from_affine(e)
        assert len(vs) <= 1, "Only a single variable can be in an affine product"
        if len(vs) == 1:
            v = list(vs)[0]
            if v.name in d:
                d[(v.name, v.uid)] = (d[(v.name, v.uid)][0] + cs, v)
            else:
                d[(v.name, v.uid)] = cs, v
        elif len(cs) > 0:
            const += prod(cs)

    # Remove var_name and negate all constants
    cks, vs = d.pop((var.name, var.uid))
    ck_val = prod(cks)

    # Aggregate all of the constants and variables
    inverse = -const / ck_val
    for cs, v in d.values():
        inverse += Const(-prod(cs) / ck_val) * v
    return inverse


def extract_moving_discontinuities(expr: ITeg,
                                   var: Var,
                                   not_ctx: Set[Tuple[str, int]],
                                   banned_variables: Set[Tuple[(str, int)]]) -> Iterable[Tuple[ITeg, ITeg]]:
    """Identify all subexpressions producing a moving discontinuity.

    Concretely, this means finding each branching statement including
    the integration variable and a variable defined in an outside context.
    """
    if isinstance(expr, IfElse):
        yield from moving_discontinuities_in_boolean(expr.cond, var, not_ctx, banned_variables)

    elif isinstance(expr, Teg):
        banned_variables.add((expr.dvar.name, expr.dvar.uid))

    yield from (moving_cond for child in expr.children
                for moving_cond in extract_moving_discontinuities(child, var, not_ctx, banned_variables))


def moving_discontinuities_in_boolean(expr: ITegBool,
                                      var: Var,
                                      not_ctx: Set[Tuple[str, int]],
                                      banned_variables: Set[Tuple[(str, int)]]) -> Iterable[ITeg]:
    if isinstance(expr, Bool):
        var_name_var_in_cond = extract_variables_from_affine(expr.left_expr - expr.right_expr)
        moving_var_name_uids = var_name_var_in_cond.keys() - not_ctx - {(var.name, var.uid)}

        # Check that the variable var is in the condition
        # and another variable not in not_ctx is in the condition
        if ((var.name, var.uid) in var_name_var_in_cond
                and len(moving_var_name_uids) > 0
                and len(banned_variables & moving_var_name_uids) == 0):
            yield expr

    elif isinstance(expr, (And, Or)):
        yield from moving_discontinuities_in_boolean(expr.left_expr, var, not_ctx, banned_variables)
        yield from moving_discontinuities_in_boolean(expr.right_expr, var, not_ctx, banned_variables)

    else:
        raise ValueError('Illegal expression in boolean.')


def delta_contribution(expr: Teg,
                       not_ctx: Set[Tuple[str, int]]
                       ) -> Dict[Tuple[str, int], Tuple[Tuple[str, int], ITeg]]:
    """Given an expression for the integral, generate an expression for the derivative of jump discontinuities. """

    # Descends into all subexpressions extracting moving discontinuities
    moving_var_data, considered_bools = [], []
    for discont_bool in extract_moving_discontinuities(expr.body, expr.dvar, not_ctx.copy(), set()):
        try:
            considered_bools.index(discont_bool)
        except ValueError:
            considered_bools.append(discont_bool)

            # Evaluate the discontinuity at x = t+
            expr_body_right = expr.body
            expr_body_right = substitute(expr_body_right, discont_bool, false)

            expr_for_dvar = solve_for_dvar(discont_bool.left_expr - discont_bool.right_expr, expr.dvar)
            expr_body_right = substitute(expr_body_right, expr.dvar, expr_for_dvar)

            # Evaluate the discontinuity at x = t-
            expr_body_left = expr.body
            expr_body_left = substitute(expr_body_left, discont_bool, true)
            expr_body_left = substitute(expr_body_left, expr.dvar, expr_for_dvar)

            # if lower < dvar < upper, include the contribution from the discontinuity (x=t+ - x=t-)
            discontinuity_happens = (expr.lower < expr_for_dvar) & (expr_for_dvar < expr.upper)
            moving_var_delta = IfElse(discontinuity_happens, expr_body_left - expr_body_right, Const(0))
            moving_var_data.append((moving_var_delta, expr_for_dvar))

    return moving_var_data

def transform_expr(expr, linear_set):
    for sub_var

def rotated_delta_contribution(expr: Teg,
                       not_ctx: Set[Tuple[str, int]]
                       ) -> Dict[Tuple[str, int], Tuple[Tuple[str, int], ITeg]]:
    """Given an expression for the integral, generate an expression for the derivative of jump discontinuities. """

    # Descends into all subexpressions extracting moving discontinuities
    moving_var_data, considered_bools = [], []
    for discont_bool in extract_moving_discontinuities(expr.body, expr.dvar, not_ctx.copy(), set()):
        try:
            considered_bools.index(discont_bool)
        except ValueError:
            considered_bools.append(discont_bool)

            # Extract rotation.
            affine_set = extract_coefficients_from_affine_tree(discont_bool.left_expr - discont_bool.right_expr)
            linear_set = affine_to_linear(affine_set)

            num_tegvars = len(linear_set)

            # Create rotated variables.
            target_vars = [TegVar(name = f'{name}_') for name, id in linear_set.keys()]

            # 

            # Evaluate the discontinuity at x = t+
            # TODO: Rewrite so it works for a general reparameterization.
            dvar = target_vars[0]
            expr_for_dvar = - constant_coefficient(affine_set)

            expr_body_right = expr.body
            expr_body_right = substitute(expr_body_right, discont_bool, false)
            expr_body_right_tx = transform_expr(expr_body_right, linear_set)
            expr_body_right_tx = substitute(expr_body_right_tx, dvar, expr_for_dvar)

            # Evaluate the discontinuity at x = t-
            expr_body_left = expr.body
            expr_body_left = substitute(expr_body_left, discont_bool, true)
            expr_body_left_tx = transform_expr(expr_body_left, linear_set)
            expr_body_left_tx = substitute(expr_body_left_tx, dvar, expr_for_dvar)

            # if lower < dvar < upper, include the contribution from the discontinuity (x=t+ - x=t-)
            lower_expr_tx = transform_expr(expr.lower, affine_set)
            upper_expr_tx = transform_expr(expr.upper, affine_set)
            dvar_tx = rotate_to_target(affine_set, target_vars = target_vars, target_index = 0)

            discontinuity_happens = (lower_expr_tx < dvar_tx) & (dvar_tx < upper_expr_tx)
            moving_var_delta = IfElse(discontinuity_happens, expr_body_left_tx - expr_body_right_tx, Const(0))

            # Build a remapping.
            remapped_delta = TegRemap(moving_var_delta, 
                                    map = dict(zip(var_list(linear_set), target_vars)),
                                    exprs = dict([
                                                (var, rotate_to_target(linear_set, 
                                                            target_vars = target_vars,
                                                            target_index = idx))
                                            for idx, var in enumerate(target_vars)]),
                                    upper_bounds = dict([
                                                (var, bounds_of(affine_set, 
                                                            target_index = idx)[1])
                                            for idx, var in enumerate(target_vars)]),
                                    lower_bounds = dict([
                                                (var, bounds_of(affine_set, 
                                                            target_index = idx)[0])
                                            for idx, var in enumerate(target_vars)])
                                )
            

            moving_var_data.append((remapped_delta, dvar_tx))

    return moving_var_data


def boundary_contribution(expr: ITeg,
                          ctx: Dict[Tuple[str, int], ITeg],
                          not_ctx: Set[Tuple[str, int]]
                          ) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """ Apply Leibniz rule directly for moving boundaries. """
    lower_deriv, ctx1, not_ctx1 = fwd_deriv_transform(expr.lower, ctx, not_ctx)
    upper_deriv, ctx2, not_ctx2 = fwd_deriv_transform(expr.upper, ctx, not_ctx)
    body_at_upper = substitute(expr.body, expr.dvar, expr.upper)
    body_at_lower = substitute(expr.body, expr.dvar, expr.lower)
    boundary_val = upper_deriv * body_at_upper - lower_deriv * body_at_lower
    return boundary_val, {**ctx1, **ctx2}, not_ctx1 | not_ctx2


def resolve_placeholders(expr: ITeg,
                         map: Dict[str, ITeg]):
    """ Substitute placeholders for their expressions """
    for key, p_expr in map:
        expr = substitute(expr, Placeholder(signature=key), p_expr)

    return expr


def fwd_deriv_transform(expr: ITeg,
                        ctx: Dict[Tuple[str, int], ITeg],
                        not_ctx: Set[Tuple[str, int]]) -> Tuple[ITeg, Dict[Tuple[str, int], str], Set[Tuple[str, int]]]:
    """Compute the source-to-source foward derivative of the given expression."""

    if isinstance(expr, Const):
        expr = Const(0)

    elif isinstance(expr, Var):
        # NOTE: No expressions with the name "d{expr.name}" are allowed
        if (expr.name, expr.uid) not in not_ctx:
            if (expr.name, expr.uid) not in ctx:
                # Introduce derivative and leave the value unbound
                var = Var(f'd{expr.name}')
                ctx[(expr.name, expr.uid)] = var
            else:
                # Use the old derivative
                var = ctx[(expr.name, expr.uid)]
            expr = var
        else:
            expr = Const(0)

    elif isinstance(expr, Add):
        def union(out1, out2):
            e1, ctx1, not_ctx1 = out1
            e2, ctx2, not_ctx2 = out2
            return expr.operation(e1, e2), {**ctx1, **ctx2}, not_ctx1 | not_ctx2
        expr, ctx, not_ctx = reduce(union, (fwd_deriv_transform(child, ctx, not_ctx) for child in expr.children))

    elif isinstance(expr, Mul):
        # NOTE: Consider n-ary multiplication.
        expr1, expr2 = [child for child in expr.children]
        deriv = [fwd_deriv_transform(child, ctx, not_ctx) for child in expr.children]
        (fwd_deriv_expr1, ctx1, not_ctx1), (fwd_deriv_expr2, ctx2, not_ctx2) = deriv
        expr = expr1 * fwd_deriv_expr2 + expr2 * fwd_deriv_expr1
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2

    elif isinstance(expr, Invert):
        deriv_expr, ctx, not_ctx = fwd_deriv_transform(expr.child, ctx, not_ctx)
        expr = -expr * expr * deriv_expr

    elif isinstance(expr, IfElse):
        if_body, ctx1, not_ctx1 = fwd_deriv_transform(expr.if_body, ctx, not_ctx)
        else_body, ctx2, not_ctx2 = fwd_deriv_transform(expr.else_body, ctx, not_ctx)
        ctx = {**ctx1, **ctx2}
        not_ctx = not_ctx1 | not_ctx2
        expr = IfElse(expr.cond, if_body, else_body)

    elif isinstance(expr, Teg):
        assert expr.dvar not in ctx, f'Names of infinitesimal "{expr.dvar}" are distinct from context "{ctx}"'
        not_ctx.discard(expr.dvar.name)

        # Include derivative contribution from moving boundaries of integration
        boundary_val, new_ctx, new_not_ctx = boundary_contribution(expr, ctx, not_ctx)
        not_ctx.add((expr.dvar.name, expr.dvar.uid))

        #moving_var_data = delta_contribution(expr, not_ctx)
        moving_var_data = rotated_delta_contribution(expr, not_ctx)
        delta_val = Const(0)
        for (moving_var_delta, expr_for_dvar) in moving_var_data:
            deriv_expr, ctx, not_ctx = fwd_deriv_transform(expr_for_dvar, ctx, not_ctx)
            delta_val += deriv_expr * moving_var_delta

        body, ctx, not_ctx = fwd_deriv_transform(expr.body, ctx, not_ctx)

        # Resolve any placeholders due to Teg bounds.
        resolve_placeholders(body, 
                                {
                                    f'{expr.dvar.id}_ub':expr.upper, 
                                    f'{expr.dvar.id}_lb':expr.lower
                                }
                            )

        ctx.update(new_ctx)
        not_ctx |= new_not_ctx
        expr = Teg(expr.lower, expr.upper, body, expr.dvar) + delta_val + boundary_val

    elif isinstance(expr, Tup):
        new_expr_list, new_ctx, new_not_ctx = [], Ctx(), set()
        for child in expr:
            child, ctx, not_ctx = fwd_deriv_transform(child, ctx, not_ctx)
            new_expr_list.append(child)
            new_ctx.update(ctx)
            new_not_ctx |= not_ctx
        ctx, not_ctx = new_ctx, new_not_ctx
        expr = Tup(*new_expr_list)

    elif isinstance(expr, LetIn):

        # Compute derivatives of each expression and bind them to the corresponding dvar
        new_vars_with_derivs, new_exprs_with_derivs = list(expr.new_vars), list(expr.new_exprs)
        for v, e in zip(expr.new_vars, expr.new_exprs):
            # By not passing in the updated contexts, require independence of exprs in the body of the let expression
            de, ctx, not_ctx = fwd_deriv_transform(e, ctx, not_ctx)
            ctx[(v.name, v.uid)] = Var(f'd{v.name}')
            new_vars_with_derivs.append(ctx[(v.name, v.uid)])
            new_exprs_with_derivs.append(de)

        # We want an expression in terms of f'd{var_in_let_body}'
        # This means that they are erroniously added to ctx, so we
        # remove them from ctx!
        dexpr, ctx, not_ctx = fwd_deriv_transform(expr.expr, ctx, not_ctx)
        [ctx.pop((c.name, c.uid), None) for c in expr.new_vars]

        expr = LetIn(Tup(*new_vars_with_derivs), Tup(*new_exprs_with_derivs), dexpr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported fwd_derivative.')

    return expr, ctx, not_ctx


def fwd_deriv(expr: ITeg, bindings: List[Tuple[ITeg, int]]) -> ITeg:
    """Computes the fwd_derivative of a given expression.

    Args:
        expr: The expression to compute the total fwd_derivative of.
        bindings: A mapping from variable names to the values of corresponding infinitesimals.

    Returns:
        Teg: The forward fwd_derivative expression.
    """
    binding_map = {(var.name, var.uid): val for var, val in bindings}

    # After fwd_deriv_transform, expr will have unbound infinitesimals
    expr, ctx, not_ctx = fwd_deriv_transform(expr, {}, set())

    # Resolve all TegRemap expressions by lifting expressions 
    # out of the tree.
    while is_remappable(expr):
        expr = remap(expr)


    assert binding_map.keys() == ctx.keys(), (f'You provided bindings for "{set(binding_map.keys())}" '
                                              f'but bindings were produced for "{set(ctx.keys())}"')

    # Bind the infinitesimals introduced by taking the derivative
    for name_uid, new_var in ctx.items():
        expr.bind_variable(new_var, binding_map[name_uid])
    return expr
