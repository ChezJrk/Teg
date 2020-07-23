from integrable_program import ITeg, Tup, Ctx
from fwd_deriv import fwd_deriv
from reverse_deriv import reverse_deriv


class FwdDeriv(ITeg):

    def __init__(self, expr: ITeg, context: Ctx):
        super(FwdDeriv, self).__init__(children=[fwd_deriv(expr, context)])
        self.deriv_expr = self.children[0]
        self.expr = expr
        self.context = context

    def __str__(self):
        return f'fwd_deriv({self.expr}, {self.context})'

    def __eq__(self, other):
        return type(self) == other and self.expr == other.expr


class RevDeriv(ITeg):

    def __init__(self, expr: ITeg, out_deriv_vals: Tup):
        super(RevDeriv, self).__init__(children=[reverse_deriv(expr, out_deriv_vals)])
        self.deriv_expr = self.children[0]
        self.expr = expr
        self.out_deriv_vals = out_deriv_vals

    def __str__(self):
        return f'reverse_deriv({self.expr}, {self.out_deriv_vals})'

    def __eq__(self, other):
        return type(self) == other and self.expr == other.expr
