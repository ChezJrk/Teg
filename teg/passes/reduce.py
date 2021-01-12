"""
Reduce to base language.
"""

from teg import (
    ITeg
)

from teg.lang.extended_utils import (
    is_base_language
)

from .delta import (
    tree_copy,
    normalize_deltas,
    eliminate_bimaps,
    eliminate_deltas
)


def reduce_to_base(expr: ITeg):
    if is_base_language(expr):
        return expr

    # Make sure tree nodes are unique
    expr = tree_copy(expr)

    # normalize all deltas.
    expr = normalize_deltas(expr)

    # eliminate all bimaps.
    expr = eliminate_bimaps(expr)

    # eliminate all deltas.
    expr = eliminate_deltas(expr)

    return expr
