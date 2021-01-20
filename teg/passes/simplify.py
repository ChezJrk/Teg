import operator
from functools import reduce

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Invert,
    SmoothFunc,
    IfElse,
    Teg,
    Tup,
    LetIn,
    BiMap,
    Or,
    And,
    Bool,
    true,
    false,
)

from teg.lang.extended import (
    Delta,
    BiMap
)

# from teg.derivs import FwdDeriv, RevDeriv

from teg.eval import evaluate as evaluate_base
from teg.passes.substitute import substitute

from functools import partial

evaluate = partial(evaluate_base, backend='numpy')


def simplify(expr: ITeg) -> ITeg:

    if isinstance(expr, Var):
        # return expr
        # if hasattr(expr, "value") and expr.value is not None:
        #    return Const(expr.value)
        return expr

    elif isinstance(expr, Add):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if isinstance(simple1, Const) and simple1.value == 0:
            return simple2
        if isinstance(simple2, Const) and simple2.value == 0:
            return simple1
        if isinstance(simple1, Const) and isinstance(simple2, Const):
            return Const(evaluate(simple1 + simple2))

        # Associative reordering.
        if isinstance(simple1, (Add, Const)) and isinstance(simple2, (Add, Const)):
            nodes1 = [simple1, ] if isinstance(simple1, Const) else simple1.children
            nodes2 = [simple2, ] if isinstance(simple2, Const) else simple2.children
            all_nodes = nodes1 + nodes2
            assert 2 <= len(all_nodes) <= 4, 'Unexpected number of nodes in Add-associative tree'

            const_nodes = [node for node in all_nodes if isinstance(node, Const)]
            other_nodes = [node for node in all_nodes if not isinstance(node, Const)]

            # No const nodes -> Reordering is pointless.
            if len(other_nodes) == len(all_nodes):
                return simple1 + simple2

            # Compress const nodes.
            const_node = Const(evaluate(reduce(operator.add, const_nodes)))

            # Re-order to front.
            if const_node == Const(0):
                simplified_nodes = other_nodes
            else:
                simplified_nodes = other_nodes + [const_node]

            # Build tree in reverse (so const node is at top level)
            return reduce(operator.add, simplified_nodes)

        if isinstance(simple1, LetIn) and isinstance(simple2, LetIn):
            if simple1.new_vars == simple2.new_vars and simple1.new_exprs == simple2.new_exprs:
                return LetIn(
                                new_vars=simple1.new_vars,
                                new_exprs=simple1.new_exprs,
                                expr=simplify(simple1.expr + simple2.expr)
                            )
            else:
                return simple1 + simple2

        if isinstance(simple1, Teg) and isinstance(simple2, Teg):
            if (simple1.dvar == simple2.dvar and simple1.lower == simple2.lower and simple1.upper == simple2.upper):
                return simplify(Teg(simple1.lower, simple1.upper, simplify(simple1.body + simple2.body), simple1.dvar))

            else:
                return simple1 + simple2

        if isinstance(simple1, IfElse) and isinstance(simple2, IfElse):
            if simple1.cond == simple2.cond:
                return IfElse(
                              simple1.cond,
                              simplify(simple1.if_body + simple2.if_body),
                              simplify(simple1.else_body + simple2.else_body)
                              )
            else:
                return simple1 + simple2

        if isinstance(simple1, Mul) and isinstance(simple2, Mul):
            # Distribution.
            # TODO: Account for second term to be t * '1'
            exprLL, exprLR = simple1.children
            exprRL, exprRR = simple2.children

            if exprLL == exprRR:
                return simplify(exprLL * (simplify(exprLR + exprRL)))
            if exprLL == exprRL:
                return simplify(exprLL * (simplify(exprLR + exprRR)))
            if exprLR == exprRL:
                return simplify(exprLR * (simplify(exprLL + exprRR)))
            if exprLR == exprRR:
                return simplify(exprLR * (simplify(exprLL + exprRL)))

        return simple1 + simple2

    elif isinstance(expr, Mul):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)

        # 0-elimination
        if ((isinstance(simple1, Const) and simple1.value == 0)
                or (isinstance(simple2, Const) and hasattr(simple2, 'value') and simple2.value == 0)):
            return Const(0)

        # Multiplicative inverse.
        if isinstance(simple1, Const) and simple1.value == 1.0:
            return simple2
        if isinstance(simple2, Const) and simple2.value == 1.0:
            return simple1

        # Local constant compression.
        if isinstance(simple1, Const) and isinstance(simple2, Const):
            return Const(evaluate(simple1 * simple2))

        # Associative reordering.
        if isinstance(simple1, (Mul, Const)) and isinstance(simple2, (Mul, Const)):
            nodes1 = [simple1] if isinstance(simple1, Const) else simple1.children
            nodes2 = [simple2] if isinstance(simple2, Const) else simple2.children
            all_nodes = nodes1 + nodes2
            assert 2 <= len(all_nodes) <= 4, 'Unexpected number of nodes in Mul-associative tree'

            const_nodes = [node for node in all_nodes if isinstance(node, Const)]
            other_nodes = [node for node in all_nodes if not isinstance(node, Const)]

            # print('const_nodes: ', const_nodes)
            # print('other_nodes: ', other_nodes)
            # No const nodes -> Reordering is pointless.
            if len(other_nodes) == len(all_nodes):
                return simple1 * simple2

            # Compress const nodes.
            # import ipdb; ipdb.set_trace()
            const_node = Const(evaluate(reduce(operator.mul, const_nodes)))

            # Re-order to front.
            if not (const_node == Const(1)):
                simplified_nodes = other_nodes + [const_node]
            else:
                simplified_nodes = other_nodes

            # print('simplified: ', simplified_nodes)

            # Build tree in reverse (so const node is at top level)
            return reduce(operator.mul, simplified_nodes)

        return simple1 * simple2

    elif isinstance(expr, Invert):
        simple = simplify(expr.child)
        if isinstance(simple, Const):
            return Const(evaluate(Invert(simple)))
        return Invert(simple)

    elif isinstance(expr, SmoothFunc):
        simple = simplify(expr.expr)
        # print(f'Simplify: {type(expr)}({type(simple)}{simple}) -> {Const(evaluate(type(expr)(simple)))}')
        if isinstance(simple, Const):
            return Const(evaluate(type(expr)(simple)))
        return type(expr)(simplify(expr.expr))

    elif isinstance(expr, IfElse):
        cond, if_body, else_body = simplify(expr.cond), simplify(expr.if_body), simplify(expr.else_body)
        if (isinstance(if_body, Const) and isinstance(else_body, Const)
                and if_body.value == 0 and else_body.value == 0):
            return if_body

        if cond == true:
            return if_body

        if cond == false:
            return else_body

        return IfElse(cond, if_body, else_body)

    elif isinstance(expr, Teg):
        body = simplify(expr.body)
        if isinstance(body, Const) and hasattr(body, 'value') and body.value == 0:
            return Const(0)
        return Teg(simplify(expr.lower), simplify(expr.upper), body, expr.dvar)

    elif isinstance(expr, Tup):
        return Tup(*(simplify(child) for child in expr))

    elif isinstance(expr, LetIn):
        # TODO: TEMP
        # if Var(name = "__norm__", uid = 26) in expr.new_vars or len(expr.new_vars) != 1:
        #    return LetIn(expr.new_vars, Tup(*(simplify(e) for e in expr.new_exprs)), simplify(expr.expr))

        simplified_exprs = Tup(*(simplify(e) for e in expr.new_exprs))
        child_expr = simplify(expr.expr)
        vars_list = expr.new_vars

        for s_var, s_expr in zip(vars_list, simplified_exprs):
            if isinstance(s_expr, Const):
                child_expr = substitute(child_expr, s_var, s_expr)

        non_const_bindings = [(s_var, s_expr) for s_var, s_expr in zip(vars_list, simplified_exprs)
                              if not isinstance(s_expr, Const)]

        child_expr = simplify(child_expr)
        if non_const_bindings:
            non_const_vars, non_const_exprs = zip(*list(non_const_bindings))
            return (LetIn(non_const_vars, non_const_exprs, child_expr)
                    if not isinstance(child_expr, Const) else child_expr)
        else:
            return child_expr

    elif isinstance(expr, BiMap):
        simplified_target_exprs = list(simplify(e) for e in expr.target_exprs)
        simplified_source_exprs = list(simplify(e) for e in expr.source_exprs)

        simplified_ubs = list(simplify(e) for e in expr.target_upper_bounds)
        simplified_lbs = list(simplify(e) for e in expr.target_lower_bounds)

        child_expr = simplify(expr.expr)

        return BiMap(expr=child_expr,
                     targets=expr.targets, target_exprs=simplified_target_exprs,
                     sources=expr.sources, source_exprs=simplified_source_exprs,
                     inv_jacobian=simplify(expr.inv_jacobian),
                     target_lower_bounds=simplified_lbs,
                     target_upper_bounds=simplified_ubs
                     )

    elif isinstance(expr, Delta):
        return Delta(simplify(expr.expr))

    # elif isinstance(expr, (FwdDeriv, RevDeriv)):
    #     return simplify(expr.deriv_expr)
    elif {'FwdDeriv', 'RevDeriv'} & {t.__name__ for t in type(expr).__mro__}:
        return simplify(expr.__getattribute__('deriv_expr'))

    elif isinstance(expr, Bool):
        left_expr, right_expr = simplify(expr.left_expr), simplify(expr.right_expr)
        if isinstance(left_expr, Const) and isinstance(right_expr, Const):
            return false if evaluate(Bool(left_expr, right_expr)) == 0.0 else true
        return Bool(left_expr, right_expr)

    elif isinstance(expr, And):
        left_expr, right_expr = simplify(expr.left_expr), simplify(expr.right_expr)
        if left_expr == true:
            return right_expr
        if right_expr == true:
            return left_expr
        if left_expr == false or right_expr == false:
            return false
        if isinstance(left_expr, Const) and isinstance(right_expr, Const):
            return Const(evaluate(And(simple1, simple2)))
        return And(left_expr, right_expr)

    elif isinstance(expr, Or):
        left_expr, right_expr = simplify(expr.left_expr), simplify(expr.right_expr)
        if left_expr == false:
            return right_expr
        if right_expr == false:
            return left_expr
        if left_expr == true or right_expr == true:
            return true
        if isinstance(left_expr, Const) and isinstance(right_expr, Const):
            return Const(evaluate(Or(simple1, simple2)))
        return Or(left_expr, right_expr)

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported simplify rule')

    """
    elif isinstance(expr, TegRemap):
        return TegRemap(
                        map=expr.map,
                        expr=simplify(expr.expr),
                        exprs=dict([(var, simplify(e)) for var, e in expr.exprs.items()]),
                        lower_bounds=dict([(var, simplify(e)) for var, e in expr.lower_bounds.items()]),
                        upper_bounds=dict([(var, simplify(e)) for var, e in expr.upper_bounds.items()]),
                        source_bounds=expr.source_bounds
                    )
    """
