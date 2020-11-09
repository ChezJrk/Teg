import operator
from functools import reduce

from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Invert,
    SmoothFunc,
    IfElse,
    Teg,
    TegRemap,
    Tup,
    LetIn,
    Or,
    And,
    Bool,
    true,
    false,
)
from derivs import FwdDeriv, RevDeriv
import operator_overloads  # noqa: F401
from evaluate import evaluate

def simplify(expr: ITeg) -> ITeg:

    if isinstance(expr, Var):
        return expr

    elif isinstance(expr, Add):
        expr1, expr2 = expr.children
        simple1, simple2 = simplify(expr1), simplify(expr2)
        if isinstance(simple1, Var) and simple1.value == 0:
            return simple2
        if isinstance(simple2, Var) and simple2.value == 0:
            return simple1
        if isinstance(simple1, Const) and isinstance(simple2, Const):
            #print(f'Simplify Add: {simple1} + {simple2}')
            return Const(evaluate(simple1 + simple2))

        # Associative reordering.
        if isinstance(simple1, (Add, Const)) and isinstance(simple2, (Add, Const)):
            nodes1 = [simple1, ] if isinstance(simple1, Const) else simple1.children
            nodes2 = [simple2, ] if isinstance(simple2, Const) else simple2.children
            all_nodes = nodes1 + nodes2
            assert 2 <= len(all_nodes) <= 4, f'Unexpected number of nodes in Add-associative tree'

            const_nodes = [node for node in all_nodes if isinstance(node, Const)]
            other_nodes = [node for node in all_nodes if not isinstance(node, Const)]
            #print('--Associative Reordering-- ')
            #print('const_nodes: ', const_nodes)
            #print('other_nodes: ', other_nodes)
            #print('all_nodes: ', all_nodes)

            # No const nodes -> Reordering is pointless.
            if len(other_nodes) == len(all_nodes):
                return simple1 + simple2

            # Compress const nodes.
            const_node = Const(evaluate(reduce(operator.add, const_nodes)))

            # Re-order to front.
            simplified_nodes = [const_node,] + other_nodes

            #print('simplified: ', simplified_nodes)

            # Build tree in reverse (so const node is at top level)
            return reduce(operator.add, simplified_nodes[::-1])

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
            assert 2 <= len(all_nodes) <= 4, f'Unexpected number of nodes in Mul-associative tree'

            const_nodes = [node for node in all_nodes if isinstance(node, Const)]
            other_nodes = [node for node in all_nodes if not isinstance(node, Const)]

            #print('const_nodes: ', const_nodes)
            #print('other_nodes: ', other_nodes)
            # No const nodes -> Reordering is pointless.
            if len(other_nodes) == len(all_nodes):
                return simple1 * simple2

            # Compress const nodes.
            const_node = Const(evaluate(reduce(operator.mul, const_nodes)))

            # Re-order to front.
            simplified_nodes = [const_node] + other_nodes
            #print('simplified: ', simplified_nodes)

            # Build tree in reverse (so const node is at top level)
            return reduce(operator.mul, simplified_nodes[::-1])

        return simple1 * simple2

    elif isinstance(expr, Invert):
        simple = simplify(expr.child)
        if isinstance(simple, Const):
            return Const(evaluate(Invert(simple)))
        return Invert(simple)

    elif isinstance(expr, SmoothFunc):
        simple = simplify(expr.expr)
        #print(f'Simplify: {type(expr)}({type(simple)}{simple}) -> {Const(evaluate(type(expr)(simple)))}')
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
        if isinstance(body, Var) and hasattr(body, 'value') and body.value == 0:
            return Const(0)
        return Teg(simplify(expr.lower), simplify(expr.upper), body, expr.dvar)

    elif isinstance(expr, Tup):
        return Tup(*(simplify(child) for child in expr))

    elif isinstance(expr, LetIn):
        return LetIn(expr.new_vars, Tup(*(simplify(e) for e in expr.new_exprs)), simplify(expr.expr))

    elif isinstance(expr, (FwdDeriv, RevDeriv)):
        return simplify(expr.deriv_expr)

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

    elif isinstance(expr, TegRemap):
        return TegRemap(
                        map = expr.map,
                        expr = simplify(expr.expr),
                        exprs = dict([(var, simplify(e)) for var, e in expr.exprs.items()]),
                        lower_bounds = dict([(var, simplify(e)) for var, e in expr.lower_bounds.items()]),
                        upper_bounds = dict([(var, simplify(e)) for var, e in expr.upper_bounds.items()]),
                        source_bounds = expr.source_bounds
                    )

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported simplify rule')
