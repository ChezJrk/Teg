import unittest
import numpy as np

from integrable_program import (
    Teg,
    TegConstant,
    TegVariable,
    TegAdd,
    TegMul,
    TegConditional,
    TegIntegral,
    TegTuple,
    TegLetIn,
)
from evaluate import evaluate
from derivs import TegFwdDeriv, TegReverseDeriv
import operator_overloads  # noqa: F401


def check_nested_lists(self, results, expected, places=7):
    for res, exp in zip(results, expected):
        if isinstance(res, (list, np.ndarray)):
            check_nested_lists(self, res, exp, places)
        else:
            t = (int, float, np.int64, np.float)
            err = f'Result {res} of type {type(res)} and expected {exp} of type {type(exp)}'
            assert isinstance(res, t) and isinstance(exp, t), err
            self.assertAlmostEqual(res, exp, places)


class TestArithmetic(unittest.TestCase):

    def test_linear(self):
        x = TegVariable('x', 1)
        three_x = x + x + x
        self.assertAlmostEqual(evaluate(three_x), 3, places=3)

    def test_multiply(self):
        x = TegVariable('x', 2)
        cube = x * x * x
        self.assertAlmostEqual(evaluate(cube), 8, places=3)

    def test_polynomial(self):
        x = TegVariable('x', 2)
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(evaluate(poly), 14, places=3)


class TestIntegrations(unittest.TestCase):

    def test_integrate_linear(self):
        a, b = 0, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_integrate_sum(self):
        a, b = 0, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} 2x dx
        integral = TegIntegral(a, b, x + x, x)
        self.assertAlmostEqual(evaluate(integral), 1, places=3)

    def test_integrate_product(self):
        a, b = -1, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x^2 dx
        integral = TegIntegral(a, b, x * x, x)
        self.assertAlmostEqual(evaluate(integral), 2 / 3, places=1)

    def test_integrate_poly(self):
        a, b = -1, 2
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, 1000), 11.25, places=3)

    def test_integrate_poly_poly_bounds(self):
        a, b = -1, 2
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a*a}^{b} x dx
        integral = TegIntegral(a * a + a + a, b * b + a + a, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, 1000), 11.25, places=3)


class TestNestedIntegrals(unittest.TestCase):

    # TODO 
    # def test_nested_integrals_with_variable_bounds(self):
    #     a, b = -1, 1
    #     x = TegVariable('x')
    #     t = TegVariable('t')
    #     a = TegVariable('a', a)
    #     b = TegVariable('b', b)
    #     # \int_-1^1 \int_{t}^{t+1} xt dx dt
    #     body = TegIntegral(t, t + b, x * t, x)
    #     integral = TegIntegral(a, b, body, t)
    #     self.assertAlmostEqual(integral.eval(100), 2 / 3, places=3)

    def test_nested_integrals_same_variable(self):
        a, b = 0, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # \int_0^1 \int_0^1 x dx dx
        body = TegIntegral(a, b, x, x)
        integral = TegIntegral(a, b, body, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_nested_integrals_different_variable(self):
        a, b = 0, 1
        x = TegVariable('x')
        t = TegVariable('t')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # \int_0^1 x * \int_0^1 t dt dx
        body = TegIntegral(a, b, t, t)
        integral = TegIntegral(a, b, x * body, x)
        self.assertAlmostEqual(evaluate(integral), 0.25, places=3)

    def test_integral_with_integral_in_bounds(self):
        a, b = 0, 1
        x = TegVariable('x')
        t = TegVariable('t')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # \int_{\int_0^1 2x dx}^{\int_0^1 xdx + \int_0^1 tdt} x dx
        integral1 = TegIntegral(a, b, x + x, x)
        integral2 = TegIntegral(a, b, t, t) + TegIntegral(a, b, x, x)

        integral = TegIntegral(integral1, integral2, x, x)
        self.assertAlmostEqual(evaluate(integral), 0, places=3)


class TestConditionals(unittest.TestCase):

    def test_basic_branching(self):
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        cond = TegConditional(x, 0, a, b)

        # if(x < c) 0 else 1 at x=-1
        cond.bind_variable('x', TegVariable('d', -1))
        self.assertEqual(evaluate(cond), 0)

        # if(x < 0) 0 else 1 at x=1
        cond.bind_variable('x', b)
        self.assertEqual(evaluate(cond, ignore_cache=True), 1)

    def test_integrate_branch(self):
        a = TegVariable('a', -1)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        d = TegVariable('d', 0)
        # if(x < c) 0 else 1
        body = TegConditional(x, 0, d, b)
        integral = TegIntegral(a, b, body, x)

        # int_{a=-1}^{b=1} (if(x < c) 0 else 1) dx
        self.assertAlmostEqual(evaluate(integral), 1, places=3)

    def test_branch_in_cond(self):
        # cond.bind_variable('x', b)
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x', -1)
        t = TegVariable('t')
        # if(x < c) 0 else 1
        upper = TegConditional(x, 0, a, b)
        integral = TegIntegral(a, upper, t, t)

        # int_{a=0}^{if(x < c) 0 else 1} t dt
        self.assertEqual(evaluate(integral), 0)


class TestTuples(unittest.TestCase):

    def test_tuple_basics(self):
        t = TegTuple(*[TegConstant(i) for i in range(5, 15)])
        two_t = t + t
        self.assertEqual(evaluate(two_t)[0], 10)

        t_squared = t * t
        self.assertEqual(evaluate(t_squared)[0], 25)

        x = TegVariable("x", 2)
        v1 = TegTuple(*[TegConstant(i) for i in range(5, 15)])
        v2 = TegTuple(*[TegConditional(x, 0, TegConstant(1), TegConstant(2))
                        for i in range(5, 15)])

        t_squared = v1 * v2
        self.assertEqual(evaluate(t_squared)[0], 10)

    def test_tuple_branch(self):
        x = TegVariable("x", 2)
        v = TegTuple(*[TegConstant(i) for i in range(5, 15)])
        cond = TegConditional(x, 0, v, TegConstant(1))
        res = TegConstant(1) + cond + v
        self.assertEqual(evaluate(res)[0], 7)

        x.bind_variable('x', -1)
        self.assertEqual(evaluate(res, ignore_cache=True)[0], 11)

    def test_tuple_integral(self):
        x = TegVariable('x')
        v = TegTuple(*[TegConstant(i) for i in range(3)])
        res = TegIntegral(TegConstant(0), TegConstant(1), v * x, x)
        res, expected = evaluate(res), [0, .5, 1]
        check_nested_lists(self, res, expected)


class TestLetIn(unittest.TestCase):

    def test_let_in(self):
        x = TegVariable('x', 2)
        y = TegVariable('y', -1)
        f = x * y
        rev_res = TegReverseDeriv(f, TegTuple(TegConstant(1)))
        # reverse_deriv(mul(x=2, y=-1), 1, 1)
        # rev_res.deriv_expr is: let ['dx=add(0, mul(1, y=-1))', 'dy=add(0, mul(1, x=2))'] in rev_deriv0 = dx, dy
        expected = [-1, 2]

        # y = -1
        fwd_res = TegFwdDeriv(f, {'x': 1, 'y': 0})
        self.assertEqual(evaluate(fwd_res), -1)

        # df/dx * [df/dx, df/dy]
        # -1 * [-1, 2] = [1, -2]
        expected = [1, -2]
        res = fwd_res * rev_res
        check_nested_lists(self, evaluate(res), expected)


class ActualIntegrationTest(unittest.TestCase):

    def test_nested(self):
        x, y = TegVariable('x', 2), TegVariable('y', -3)
        v = TegTuple(*[TegFwdDeriv(x + y, {'x': 1, 'y': 0}), TegReverseDeriv(x * y, TegTuple(TegConstant(1)))])

        expected = [1, [-3, 2]]
        check_nested_lists(self, evaluate(v), expected)

        x.unbind_variable('x')
        # \int_0^1 [1, [-3, 2]] * x dx = [\int_0^1 x, [\int_0^1 -3x, \int_0^1 x^2]] = [1/2, [-3/2, 1/3]] 
        res = TegIntegral(TegConstant(0), TegConstant(1), v * x, x)
        expected = [1/2, [-3/2, 1/3]]
        check_nested_lists(self, evaluate(res), expected, places=3)


if __name__ == '__main__':
    unittest.main()
