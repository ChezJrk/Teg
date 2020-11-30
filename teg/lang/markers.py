from typing import Optional, Set, Dict, Tuple
from .integrable_program import (
    ITeg,
    Var,
    TegVar
)


class Placeholder(Var):
    """
        Replaceable tag.
    """
    def __init__(self, name: str = '', signature: str = ''):
        super(Placeholder, self).__init__(name=name)
        self.signature = signature

    def bind_variable(self, var: Var, value: Optional[float] = None) -> None:
        if (self.name, self.uid) == (var.name, var.uid):
            self.value = value


class TegRemap(ITeg):
    """
        Intermediate element that holds a variable remapping as well as target expressions.
    """
    def __init__(self,
                 expr: ITeg,
                 map: Dict[Tuple[str, int], Tuple[str, int]],
                 exprs: Dict[Tuple[str, int], ITeg],
                 upper_bounds: Dict[Tuple[str, int], ITeg],
                 lower_bounds: Dict[Tuple[str, int], ITeg],
                 source_bounds: Set[Tuple[TegVar, ITeg, ITeg]],
                 name: str = 'TegRemap'):
        super(TegRemap, self).__init__(children=[expr])
        self.map = map
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.source_bounds = source_bounds
        self.expr = expr
        self.exprs = exprs
        self.operation = None  # Cannot eval.
        self.name = name
