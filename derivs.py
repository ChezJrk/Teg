from integrable_program import Teg, TegContext
from fwd_deriv import fwd_deriv
from reverse_deriv import reverse_deriv


class Derivative(Teg):

    def __init__(self, expr: Teg, context: TegContext, sign=1):
        super(Derivative, self).__init__(children=[], sign=sign)
        self.children[0] = self.operation(self.children[0], self.context)
        self.deriv_expr = self.children[0]
        self.expr = expr
        self.context = context

    def __str__(self):
        return f'{self.name}({self.expr}, {self.context}, {self.sign})'

    def eval(self, num_samples: int = 50, ignore_cache: bool = False) -> float:
        if self.value is None or ignore_cache:
            self.value = self.deriv_expr.eval(num_samples, ignore_cache)
        return self.value


class TegFwdDeriv(Derivative):
    name = 'fwd_deriv'
    operation = fwd_deriv


class TegReverseDeriv(Derivative):
    name = 'reverse_deriv'
    operation = reverse_deriv
