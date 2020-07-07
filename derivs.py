from integrable_program import Teg, TegTuple, TegContext
from fwd_deriv import fwd_deriv
from reverse_deriv import reverse_deriv


class TegFwdDeriv(Teg):

    def __init__(self, expr: Teg, context: TegContext):
        super(TegFwdDeriv, self).__init__(children=[fwd_deriv(expr, context)])
        self.deriv_expr = self.children[0]
        self.expr = expr
        self.context = context

    def __str__(self):
        return f'fwd_deriv({self.expr}, {self.context}, {self.sign})'


class TegReverseDeriv(Teg):

    def __init__(self, expr: Teg, out_deriv_vals: TegTuple):
        super(TegReverseDeriv, self).__init__(children=[reverse_deriv(expr, out_deriv_vals)])
        self.deriv_expr = self.children[0]
        self.expr = expr
        self.out_deriv_vals = out_deriv_vals

    def __str__(self):
        return f'reverse_deriv({self.expr}, {self.out_deriv_vals}, {self.sign})'
