"""
Reduce to base language.
"""
import time

from teg import ITeg
from teg.lang.extended_utils import is_base_language
from .delta import tree_copy, normalize_deltas, eliminate_bimaps, eliminate_deltas
from .simplify import simplify


def reduce_to_base(expr: ITeg, timing=False):
    if is_base_language(expr):
        return expr

    # Make sure tree nodes are unique
    expr = tree_copy(expr)

    # normalize all deltas.
    start = time.time()
    expr = simplify(normalize_deltas(expr))
    end = time.time()
    if timing:
        print(f'\tDelta normalization: {end - start:.2f}s')

    # eliminate all bimaps.
    start = time.time()
    expr = simplify(eliminate_bimaps(expr))
    end = time.time()
    if timing:
        print(f'\tBimap elimination: {end - start:.2f}s')

    # eliminate all deltas.
    start = time.time()
    expr = simplify(eliminate_deltas(expr))
    end = time.time()
    if timing:
        print(f'\tDelta elimination: {end - start:.2f}s')

    return expr
