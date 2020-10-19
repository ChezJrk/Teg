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

    remap_expr, parent_expr, teg_list = remap_gather(expr)
    if remap_expr is None:
        return expr

    assert is_remappable(parent_expr) == False, f'Remapped expression is not a linear subtree in the provided tree'

    expr = substitute(expr, remap_expr, Const(0))

    new_expr = parent_expr
    for integral in teg_list:
        dvar, lexpr, uexpr = integral
        new_expr = Teg(new_expr, dvar, lexpr, uexpr)

    expr = expr + new_expr

    return expr

def remap_gather(expr: ITeg):
    if isinstance(expr, TegRemap):
        return expr, expr, []

    elif isinstance(expr, Teg):
        remap_expr, parent_expr, teg_list = remap_gather(expr.body)
        if remap_expr is None:
            return None, None, []

        # Lookup remapped variable.
        if (expr.dvar.name, expr.dvar.id) not in remap_expr.map:
            parent_expr = Teg(parent_expr, expr.dvar, expr.lower, expr.upper) # TODO: Double check positions
            return remap_expr, parent_expr, teg_list
        else:
            new_name, new_id = remap_expr.map[(expr.dvar.name, expr.name.id)]

            lexpr = remap_expr.lower_bounds[(new_name, new_id)]
            uexpr = remap_expr.upper_bounds[(new_name, new_id)]

            new_dvar = TegVar(name = new_name, id = new_id)
            parent_expr = Teg(parent_expr, new_dvar, lexpr, uexpr)

            return remap_expr, parent_expr, teg_list + [(new_dvar, lexpr, uexpr)]

    elif isinstance(expr, LetIn):
        # TODO: Add exprs here.
        # Do the same thing here.
        # Try an example first.
        pass

    elif isinstance(expr, Add):
        for child in expr.children: 
            remap_expr, parent_expr, teg_list = remap_gather(child)
            if remap_expr is None:
                continue
            # Ignore the 'add' operation.
            return remap_expr, parent_expr, teg_list

        return None, None, []

    elif isinstance(expr, Mul):
        for index, child in enumerate(expr.children): 
            remap_expr, parent_expr, teg_list = remap_gather(child)
            if remap_expr is None:
                continue

            # Retain the 'mul' operation, but replace this child with the pruned tree 'parent_expr'
            return remap_expr, 
                    Mul(children = [parent_expr] + [child for i, child in enumerate(expr.children) if i != index]),
                    teg_list

        return None, None, []

    elif isinstance(expr, ITeg):
        

    return expr