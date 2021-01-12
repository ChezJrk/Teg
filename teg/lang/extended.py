from typing import Tuple
from .base import (
    ITeg,
    Var
)


class ITegExtended(ITeg):
    def __init__(self, *args, **kwargs):
        super(ITegExtended, self).__init__(*args, **kwargs)
        pass


class BiMap(ITegExtended):

    def __init__(self,
                 expr: ITeg,
                 targets: Tuple[Var],
                 target_exprs: Tuple[ITeg],
                 sources: Tuple[Var],
                 source_exprs: Tuple[ITeg] = None,
                 inv_jacobian: ITeg = None,
                 target_upper_bounds: ITeg = None,
                 target_lower_bounds: ITeg = None):
        super(BiMap, self).__init__(children=[expr,
                                              *target_exprs, *source_exprs,
                                              inv_jacobian,
                                              *target_lower_bounds, *target_upper_bounds])
        self.expr = expr
        self.targets = targets
        self.target_exprs = target_exprs
        self.sources = sources
        self.source_exprs = source_exprs  # optional
        self.inv_jacobian = inv_jacobian  # optional
        self.target_upper_bounds = target_upper_bounds  # optional
        self.target_lower_bounds = target_lower_bounds  # optional


class Delta(ITegExtended):

    def __init__(self,
                 expr: ITeg):
        super(Delta, self).__init__(children=[expr])

        self.expr = expr
