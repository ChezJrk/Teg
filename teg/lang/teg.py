from typing import Optional

from .base import Var, ITeg, try_making_teg_const
from .markers import Placeholder


class TegVar(Var):
    def __init__(self, name: str = '', uid: Optional[int] = None):
        super(TegVar, self).__init__(name=name, uid=uid)

    def bind_variable(self, var: Var, value: Optional[float] = None) -> None:
        if (self.name, self.uid) == (var.name, var.uid):
            self.value = value

    def upper_bound(self):
        return Placeholder(signature=f'{self.uid}_ub')

    def lower_bound(self):
        return Placeholder(signature=f'{self.uid}_lb')

    def ub(self):
        """ Shorthand for upper_bound() """
        return self.upper_bound()

    def lb(self):
        """ Shorthand for lower_bound() """
        return self.lower_bound()


class Teg(ITeg):

    def __init__(self, lower: Var, upper: Var, body: ITeg, dvar: TegVar):
        super(Teg, self).__init__(children=[try_making_teg_const(e) for e in (lower, upper, body)])
        self.lower, self.upper, self.body = self.children
        assert isinstance(dvar, TegVar), f'Can only integrate over TegVar variables. {dvar} is not a TegVar'
        self.dvar = dvar
