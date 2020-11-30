import unittest

from integrable_program import Teg, Var, IfElse, Const, Tup, LetIn
from fwd_deriv import fwd_deriv
from reverse_deriv import reverse_deriv
from evaluate import evaluate
import operator_overloads  # noqa: F401


class TestForwardDerivatives(unittest.TestCase):

    def test_deriv_basics(self):
        x = Var('x', 1)
        f = x + x
        deriv_expr = fwd_deriv(f, [(x, 1)])
        # df(x=1)/dx
        self.assertEqual(evaluate(deriv_expr), 2)

        f = x * x
        # df(x=1)/dx
        deriv_expr = fwd_deriv(f, [(x, 1)])
        self.assertEqual(evaluate(deriv_expr), 2)

    def test_deriv_poly(self):
        x = Var('x', 1)
        f = 1 * x**3 + 3 * x**2 + 3 * x + 1
        # df(x=1)/dx
        deriv_expr = fwd_deriv(f, [(x, 1)])
        self.assertEqual(evaluate(deriv_expr), 12)

        deriv_expr = fwd_deriv(f, [(x, 1)])
        deriv_expr.bind_variable(x, -1)
        # df(x=-1)/dx
        self.assertEqual(evaluate(deriv_expr, ignore_cache=True), 0)

    def test_deriv_branch(self):
        x = Var('x')
        theta = Var('theta', 1)
        f = IfElse(x < 0, theta, theta * theta)

        deriv_expr = fwd_deriv(f, [(theta, 1)])
        x.bind_variable(x, 1)

        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(evaluate(deriv_expr), 2)

        x.bind_variable(x, -1)
        self.assertAlmostEqual(evaluate(deriv_expr, ignore_cache=True), 1)

    def test_deriv_integral(self):
        a = Const(name='a', value=0)
        b = Const(name='b', value=1)

        x = Var('x')
        theta = Var('theta', 5)
        f = Teg(a, b, x * theta, x)

        deriv_expr = fwd_deriv(f, [(theta, 1)])

        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(evaluate(deriv_expr), 0.5)

    def test_deriv_integral_branch_poly(self):
        a = Const(name='a', value=0)
        b = Const(name='b', value=1)

        theta1 = Var('theta1', -1)
        theta2 = Var('theta2', 1)
        # g = \int_a^b if(x<0.5) x*theta1 else 1 dx
        # h = \int_a^b x*theta2 + x^2theta1^2 dx
        x = Var('x')
        body = IfElse(x < Const(0.5), x * theta1, b)
        g = Teg(a, b, body, x)
        h = Teg(a, b, x * theta2 + x**2 * theta1**2, x)

        y = Var('y', 0)
        # if(y < 1) g else h
        f = IfElse(y < Const(1), g, h)

        # df/d(theta1)
        deriv_expr = fwd_deriv(f, [(theta1, 1), (theta2, 0)])
        self.assertAlmostEqual(evaluate(deriv_expr), 0.125, places=3)

        # df/d(theta2)
        deriv_expr = fwd_deriv(f, [(theta1, 0), (theta2, 1)])
        self.assertAlmostEqual(evaluate(deriv_expr, ignore_cache=True), 0)

        f.bind_variable(y, 2)
        deriv_expr = fwd_deriv(f, [(theta1, 1), (theta2, 0)])
        self.assertAlmostEqual(evaluate(deriv_expr, ignore_cache=True), -2/3, places=3)

        deriv_expr = fwd_deriv(f, [(theta1, 0), (theta2, 1)])
        self.assertAlmostEqual(evaluate(deriv_expr, ignore_cache=True), 1/2, places=3)

    def test_deriv_tuple(self):
        x = Var('x', 1)
        y = Var('y', 1)
        t = Var('t')
        a = Const(0)
        b = Const(1)

        cond = IfElse(x < a, x, y)
        integral = Teg(a, b, t * x, t)

        f = Tup(x, x + x, x * x, x + y, x * y, cond, integral)
        deriv_expr = fwd_deriv(f, [(x, 1), (y, 0)])

        expected = [1, 2, 2, 1, 1, 0, 0.5]
        for res, exp in zip(evaluate(deriv_expr), expected):
            self.assertAlmostEqual(res, exp)

        expected = [0, 0, 0, 1, 1, 1, 0]
        deriv_expr = fwd_deriv(f, [(x, 0), (y, 1)])
        for res, exp in zip(evaluate(deriv_expr), expected):
            self.assertAlmostEqual(res, exp)

    def test_deriv_let_in(self):
        x1 = Var('x1')
        x2 = Var('x2')

        y = Var('y', 1)
        z = Var('z', 2)

        # let x1 = y + z
        #     x2 = y * z in
        # w = x1 + x2 * y
        new_vars = Tup(x1, x2)
        new_exprs = Tup(y + z, y * z)

        expr = x1 + x2 * y
        letin = LetIn(new_vars, new_exprs, expr)
        deriv_expr = fwd_deriv(letin, [(y, 0), (z, 1)])
        self.assertEqual(evaluate(deriv_expr), 2)

        deriv_expr = fwd_deriv(letin, [(y, 1), (z, 0)])
        self.assertEqual(evaluate(deriv_expr, ignore_cache=True), 5)


class TestReverseDerivatives(unittest.TestCase):

    def setUp(self):
        self.single_out_deriv = Tup(Const(1))

    def test_deriv_basics(self):
        x = Var('x', 1)
        f = x + x
        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        # df(x=1)/dx
        self.assertEqual(evaluate(deriv_expr), 2)

        f = x * x
        # # df(x=1)/dx
        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        self.assertEqual(evaluate(deriv_expr), 2)

    def test_deriv_poly(self):
        x = Var('x', 1)
        c1 = Const(1)
        c2 = Const(3)
        f = c1 * x**3 + c2 * x**2 + c2 * x + c1
        # df(x=1)/dx
        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        self.assertEqual(evaluate(deriv_expr), 12)

        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        deriv_expr.bind_variable(x, -1)
        # df(x=-1)/dx
        self.assertEqual(evaluate(deriv_expr, ignore_cache=True), 0)

    def test_deriv_branch(self):
        x = Var('x')
        theta = Var('theta', 1)
        f = IfElse(x < 0, theta, theta * theta)

        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        x.bind_variable(x, 1)
        # deriv(if (x<0) theta else theta^2)
        self.assertAlmostEqual(evaluate(deriv_expr), 2)

        x.bind_variable(x, -1)
        self.assertAlmostEqual(evaluate(deriv_expr, ignore_cache=True), 1)

    def test_deriv_integral(self):
        a = Const(name='a', value=0)
        b = Const(name='b', value=1)

        x = Var('x')
        theta = Var('theta', 5)
        f = Teg(a, b, x * theta, x)

        deriv_expr = reverse_deriv(f, self.single_out_deriv)

        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(evaluate(deriv_expr), 0.5)

    def test_deriv_integral_branch_poly(self):
        a = Const(name='a', value=0)
        b = Const(name='b', value=1)

        theta1 = Var('theta1', -1)
        theta2 = Var('theta2', 1)
        # g = \int_a^b if(x<0.5) x*theta1 else 1 dx
        # h = \int_a^b x*theta2 + x^2theta1^2 dx
        x = Var('x')
        body = IfElse(x < Const(0.5), x * theta1, b)
        g = Teg(a, b, body, x)
        h = Teg(a, b, x * theta2 + x**2 * theta1**2, x)

        y = Var('y', 0)
        # if(y < 1) g else h
        f = IfElse(y < Const(1), g, h)

        # [df/d(theta1), df/d(theta2)]
        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        df_dtheta1, df_dtheta2 = evaluate(deriv_expr)
        self.assertAlmostEqual(df_dtheta1, 0.125, places=3)
        self.assertAlmostEqual(df_dtheta2, 0)

        f.bind_variable(y, 2)
        deriv_expr = reverse_deriv(f, self.single_out_deriv)
        df_dtheta1, df_dtheta2 = evaluate(deriv_expr, ignore_cache=True)
        self.assertAlmostEqual(df_dtheta1, -2/3, places=3)
        self.assertAlmostEqual(df_dtheta2, 1/2, places=3)

    def test_deriv_tuple(self):
        x = Var('x', -1)
        y = Var('y', 2)
        t = Var('t')
        a = Const(0)
        b = Const(1)

        cond = IfElse(x < a, x, y)
        integral = Teg(a, b, t * x, t)

        f = Tup(x, x + x, x * x, x + y, x * y, cond, integral)
        out_derivs = Tup(*[Const(1) for i in range(len(f.children))])
        deriv_expr = reverse_deriv(f, out_derivs)

        expected = [1, 2, -2, [1, 1], [2, -1], [1, 0], 0.5]
        for res, exp in zip(evaluate(deriv_expr), expected):
            if isinstance(exp, list):
                for inner_res, inner_exp in zip(res, exp):
                    self.assertAlmostEqual(inner_res, inner_exp)
            else:
                self.assertAlmostEqual(res, exp)

    def test_deriv_let_in(self):
        x1 = Var('x1')
        x2 = Var('x2')

        y = Var('y', 1)
        z = Var('z', 2)

        # let x1 = y + z
        #     x2 = y * z in
        # x1 + x2 * y
        new_vars = Tup(x1, x2)
        new_exprs = Tup(y + z, y * z)

        expr = x1 + x2 * y
        letin = LetIn(new_vars, new_exprs, expr)

        out_derivs = Tup(Const(1))
        deriv_expr = reverse_deriv(letin, out_derivs)

        expected = [5, 2]
        for r, e in zip(evaluate(deriv_expr), expected):
            self.assertEqual(r, e)

# TODO: Factor out all shared functions to distill testing code
# TODO: Implement derivatives
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) x
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) t


if __name__ == '__main__':
    unittest.main()
