"""
    Contains methods that perform variable remapping
    on Teg expression trees.
"""

from typing import Dict, Set, List, Tuple, Iterable
from functools import reduce
from itertools import product
import operator

from integrable_program import (
    ITeg,
    Const,
    Var,
    TegVar,
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

from integrable_program import (
    Placeholder,
    TegRemap,
)

from substitute import substitute


def is_remappable(expr: ITeg):
    return remap_gather(expr) is not None

def remap(expr: ITeg):
    """
        Performs a remap pass. 
        Eliminates 'TegRemap' nodes by lifting the subtree to the top level
        of the tree and applying variable rewrites to them.
    """

    remap_expr, remapped_tree, teg_list = remap_gather(expr)
    if remap_expr is None:
        return expr

    assert is_remappable(remapped_tree) == False, f'Remapped expression is not a linear subtree in the provided tree'

    expr = substitute(expr, remap_expr, Const(0))

    new_expr = remapped_tree
    for integral in teg_list:
        dvar, lexpr, uexpr = integral
        new_expr = Teg(new_expr, dvar, lexpr, uexpr)

    expr = expr + new_expr

    return expr

def remap_gather(expr: ITeg):
    if isinstance(expr, TegRemap):
        return expr, expr, []

    elif isinstance(expr, Teg):
        remap_expr, remapped_tree, teg_list = remap_gather(expr.body)
        if remap_expr is None:
            return None, None, []

        # Lookup remapped variable.
        if (expr.dvar.name, expr.dvar.id) not in remap_expr.map:
            remapped_tree = Teg(remapped_tree, expr.dvar, expr.lower, expr.upper) # TODO: Double check positions
            return remap_expr, remapped_tree, teg_list
        else:
            new_name, new_id = remap_expr.map[(expr.dvar.name, expr.name.id)]

            lexpr = remap_expr.lower_bounds[(new_name, new_id)]
            uexpr = remap_expr.upper_bounds[(new_name, new_id)]

            new_dvar = TegVar(name = new_name, id = new_id)
            #remapped_tree = Teg(remapped_tree, new_dvar, lexpr, uexpr)

            return remap_expr, remapped_tree, teg_list + [(new_dvar, lexpr, uexpr)]

    elif isinstance(expr, LetIn):
        for idx, child in enumerate(expr.children):
            remap_expr, remapped_tree, teg_list = remap_gather(child)
            if remap_expr is not None:
                children = expr.children[:idx] + [remapped_tree] + expr.children[idx + 1:]
                return remap_expr, LetIn(new_vars = expr.new_vars, new_exprs = children[1:], expr = children[0])

        return None, None, []

    elif isinstance(expr, Add):
        for child in expr.children: 
            remap_expr, remapped_tree, teg_list = remap_gather(child)
            if remap_expr is None:
                continue
            # Ignore the 'add' operation.
            return remap_expr, remapped_tree, teg_list

        return None, None, []

    elif isinstance(expr, Mul):
        for index, child in enumerate(expr.children): 
            remap_expr, remapped_tree, teg_list = remap_gather(child)
            if remap_expr is None:
                continue

            # Retain the 'mul' operation, but replace this child with the pruned tree 'remapped_tree'
            return (remap_expr, 
                    Mul(children = expr.children[:index] + [remapped_tree] + expr.children[index+1:]), 
                    teg_list)

        return None, None, []
    
    elif isinstance(expr, Invert):
        remap_expr, remapped_tree, teg_list = remap_gather(expr.child)
        return (remap_expr,
               Invert(remapped_tree),
               teg_list)

    elif isinstance(expr, Tup):
        for index, child in enumerate(expr.children): 
            remap_expr, remapped_tree, teg_list = remap_gather(child)
            if remap_expr is None:
                continue

            # Retain the 'mul' operation, but replace this child with the pruned tree 'remapped_tree'
            return (remap_expr, 
                    Tup(expr.children[:index] + [remapped_tree] + expr.children[index + 1:]),
                    teg_list)
        return None, None, []

    elif isinstance(expr, (Bool, And, Or)):
        remap_expr, remapped_tree, teg_list = remap_gather(expr.left_expr)
        if remap_expr is not None:
            return remap_expr, type(expr) (remapped_tree, expr.right_expr), teg_list

        remap_expr, remapped_tree, teg_list = remap_gather(expr.right_expr)
        if remap_expr is not None:
            return remap_expr, type(expr) (expr.left_expr, remapped_tree), teg_list
        
        return None, None, []

    elif isinstance(expr, IfElse):
        assert is_remappable(expr.cond) == False, "Can't have remaps within conditions."

        remap_expr, remapped_tree, teg_list = remap_gather(expr.if_body)
        if remap_expr is not None:
            return remap_expr, IfElse(cond, remapped_tree, expr.else_body), teg_list

        remap_expr, remapped_tree, teg_list = remap_gather(expr.else_body)
        if remap_expr is not None:
            return remap_expr, IfElse(cond, expr.if_body, remapped_tree), teg_list
        
        return None, None, []

    elif isinstance(expr, (Var, Const, TegVar)):
        return None, None, []

    else:
        # Warning for future proofing.
        print("WARNING: Remap traversal doesn't support this type")
        return None, None, []

    return None, None, []