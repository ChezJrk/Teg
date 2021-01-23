from typing import List, Optional
from teg import (
    ITeg,
    Tup,
    Ctx,
    Var
)

from .fwd_deriv import fwd_deriv
from .reverse_deriv import reverse_deriv

from teg.passes.reduce import reduce_to_base


class FwdDeriv(ITeg):

    def __init__(self, expr: ITeg, context: Ctx):
        super(FwdDeriv, self).__init__(children=[fwd_deriv(expr, context)])
        self.deriv_expr = self.children[0]
        # print('Original')
        # print(self.deriv_expr)
        self.deriv_expr = reduce_to_base(self.deriv_expr)
        self.expr = expr
        self.context = context

    def __str__(self):
        return f'fwd_deriv({self.expr}, {self.context})'

    def __eq__(self, other):
        return type(self) == other and self.expr == other.expr


class RevDeriv(ITeg):

    def __init__(self, expr: ITeg, out_deriv_vals: Tup, output_list: Optional[List[Var]] = None):
        super(RevDeriv, self).__init__(children=[])
        variables, deriv_expr = reverse_deriv(expr, out_deriv_vals, output_list=output_list)
        # print(variables)
        # print(deriv_expr)
        deriv_expr = reduce_to_base(deriv_expr)

        self.variables = variables
        self.children = [deriv_expr]
        self.deriv_expr = deriv_expr
        self.expr = expr
        self.out_deriv_vals = out_deriv_vals

    def __str__(self):
        return f'reverse_deriv({self.expr}, {self.out_deriv_vals})'

    def __eq__(self, other):
        return type(self) == other and self.expr == other.expr
