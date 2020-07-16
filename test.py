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
from simplify import simplify

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
        cond = TegConditional(x, TegConstant(0), a, b)

        # if(x < c) 0 else 1 at x=-1
        cond.bind_variable('x', -1)
        self.assertEqual(evaluate(cond), 0)

        # if(x < 0) 0 else 1 at x=1
        cond.bind_variable('x', 1)
        self.assertEqual(evaluate(cond, ignore_cache=True), 1)

    def test_integrate_branch(self):
        a = TegVariable('a', -1)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        d = TegVariable('d', 0)
        # if(x < c) 0 else 1
        body = TegConditional(x, TegConstant(0), d, b)
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
        upper = TegConditional(x, TegConstant(0), a, b)
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
        v2 = TegTuple(*[TegConditional(x, TegConstant(0), TegConstant(1), TegConstant(2))
                        for i in range(5, 15)])

        t_squared = v1 * v2
        self.assertEqual(evaluate(t_squared)[0], 10)

    def test_tuple_branch(self):
        x = TegVariable("x", 2)
        v = TegTuple(*[TegConstant(i) for i in range(5, 15)])
        cond = TegConditional(x, TegConstant(0), v, TegConstant(1))
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

    def test_deriv_inside_integral(self):
        x = TegVariable('x')

        integral = TegIntegral(TegConstant(0), TegConstant(1), TegFwdDeriv(x * x, {'x': 1}), x)
        self.assertAlmostEqual(evaluate(integral), 1)

        integral = TegIntegral(TegConstant(0), TegConstant(1), TegReverseDeriv(x * x, TegTuple(TegConstant(1))), x)
        self.assertAlmostEqual(evaluate(integral, ignore_cache=True), 1)

    def test_deriv_outside_integral(self):
        x = TegVariable('x')
        integral = TegIntegral(TegConstant(0), TegConstant(1), x, x)

        with self.assertRaises(AssertionError):
            d_integral = TegFwdDeriv(integral, {'x': 1})
            evaluate(d_integral)


class VariableBranchConditionsTest(unittest.TestCase):

    def test_deriv_heaviside(self):
        x = TegVariable('x')
        t = TegVariable('t', 0.5)
        heaviside = TegConditional(x, t, TegConstant(0), TegConstant(1))
        integral = TegIntegral(TegConstant(0), TegConstant(1), heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 1)

        neg_heaviside = TegConditional(x, t, TegConstant(1), TegConstant(0))
        integral = TegIntegral(TegConstant(0), TegConstant(1), neg_heaviside, x)

        # d/dt int_{x=0}^{1} (x<t) ? 1 : 0
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), -1)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), -1)

    def test_deriv_heaviside_discontinuity_out_of_domain(self):
        x = TegVariable('x')
        t = TegVariable('t', 2)
        heaviside = TegConditional(x, t, TegConstant(0), TegConstant(1))
        integral = TegIntegral(TegConstant(0), TegConstant(1), heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

        t.bind_variable('t', -1)
        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 0)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

    def test_deriv_scaled_heaviside(self):
        x = TegVariable('x')
        t = TegVariable('t', 0.5)
        heavyside_t = TegConditional(x, t, TegConstant(0), TegConstant(1))
        body = (x + t) * heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 (x + t) * ((x<t) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1.5)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 1.5)

    def test_deriv_add_heaviside(self):
        x = TegVariable('x')
        t = TegVariable('t', 0.5)
        heavyside_t = TegConditional(x, t, TegConstant(0), TegConstant(1))
        body = x + t + heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 x + t + ((x<t) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 2)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 2)

    def test_deriv_sum_heavisides(self):
        x = TegVariable('x')
        t = TegVariable('t', 0.5)
        heavyside_t = TegConditional(x, t, TegConstant(0), TegConstant(1))
        flipped_heavyside_t = TegConditional(x, t, TegConstant(1), TegConstant(0))
        body = flipped_heavyside_t + TegConstant(2) * heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 1 : 0) + 2 * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 1)

        t1 = TegVariable('t1', 0.3)
        heavyside_t1 = TegConditional(x, t1, TegConstant(0), TegConstant(1))
        body = heavyside_t + heavyside_t1
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) + ((x<t=0.3) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1, 't1': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 2)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), [1, 1])

        body = heavyside_t + heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) + ((x<t=0.3) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 2)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 2)

    def test_deriv_product_heavisides(self):
        x = TegVariable('x')
        t = TegVariable('t', 0.5)
        heavyside_t = TegConditional(x, t, TegConstant(0), TegConstant(1))
        flipped_heavyside_t = TegConditional(x, t, TegConstant(1), TegConstant(0))
        body = heavyside_t * heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 1)

        body = flipped_heavyside_t * heavyside_t
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 1 : 0) * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 0)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

        t1 = TegVariable('t1', 0.3)
        heavyside_t1 = TegConditional(x, t1, TegConstant(0), TegConstant(1))
        body = heavyside_t * heavyside_t1
        integral = TegIntegral(TegConstant(0), TegConstant(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) * ((x<t1=0.3) ? 0 : 1)
        fwd_dintegral = TegFwdDeriv(integral, {'t': 1, 't1': 1})
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 1)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), [1, 0])

    def test_tzu_mao(self):
        # \int_{x=0}^1 x < t ?
        #   \int_{y=0}^1 x < t1 ?
        #     x * y : x * y^2 :
        #   \int_{y=0}^1 y < t2 ?
        #     x^2 * y : x^2 * y^2
        zero, one = TegConstant(0), TegConstant(1)
        x, y = TegVariable('x'), TegVariable('y')
        t, t1, t2 = TegVariable('t', 0.5), TegVariable('t1', 0.25), TegVariable('t2', 0.75)
        if_body = TegIntegral(zero, one, TegConditional(x, t1, x * y, x * y * y), y)
        else_body = TegIntegral(zero, one, TegConditional(y, t2, x * x * y, x * x * y * y), y)
        integral = TegIntegral(zero, one, TegConditional(x, t, if_body, else_body), x)

        rev_dintegral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        expected = [-0.048, -0.041, -0.054]
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), expected, places=2)

        fwd_dintegral = TegFwdDeriv(integral, {'t': 1, 't1': 0, 't2': 0})
        dt = evaluate(fwd_dintegral, ignore_cache=True)

        fwd_dintegral = TegFwdDeriv(integral, {'t': 0, 't1': 1, 't2': 0})
        dt1 = evaluate(fwd_dintegral, ignore_cache=True)

        fwd_dintegral = TegFwdDeriv(integral, {'t': 0, 't1': 0, 't2': 1})
        dt2 = evaluate(fwd_dintegral, ignore_cache=True)

        check_nested_lists(self, [dt, dt1, dt2], expected, places=2)

    def test_single_integral_example(self):
        # deriv(\int_{x=0}^1 (x < theta ? 1 : x * theta))
        zero, one = TegConstant(0), TegConstant(1)
        x, theta = TegVariable('x'), TegVariable('theta', 0.5)

        body = TegConditional(x, theta, one, x * theta)
        integral = TegIntegral(zero, one, body, x)
        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -0.375, places=2)

    def test_double_integral(self):
        # \int_{x=0}^1
        #   deriv(
        #       \int_{y=0}^1
        #           (x < y ? 0 : 1)
        zero, one = TegConstant(0), TegConstant(1)
        x, y = TegVariable('x'), TegVariable('y')

        body = TegConditional(x, y, zero, one)
        integral = TegIntegral(zero, one, body, x)
        body = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        double_integral = TegIntegral(zero, one, body, y)
        self.assertAlmostEqual(evaluate(double_integral, num_samples=100), 1, places=1)

    def test_nested_integral_moving_discontinuity(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = TegConstant(0), TegConstant(1)
        x, y, t = TegVariable('x'), TegVariable('y'), TegVariable('t', 0.5)

        body = TegIntegral(zero, one, TegConditional(x, t, zero, one), x)
        integral = TegIntegral(zero, one, y * body, y)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 0.5, places=3)

        deriv_integral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 0.5, places=3)


class MovingBoundaryTest(unittest.TestCase):

    def test_moving_upper_boundary(self):
        # deriv(\int_{x=0}^{t=1} xt)
        # 0.5 + 1 - 0 = 1.5
        zero = TegConstant(0)
        x, t = TegVariable('x'), TegVariable('t', 1)
        integral = TegIntegral(zero, t, x * t, x)

        deriv_integral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(deriv_integral), 1.5)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 1.5)

    def test_moving_lower_boundary(self):
        # deriv(\int_{x=t=-1}^{1} xt)
        # 0 + 0 + 1 = 1
        one = TegConstant(1)
        x, t = TegVariable('x'), TegVariable('t', -1)
        integral = TegIntegral(t, one, x * t, x)

        deriv_integral = TegFwdDeriv(integral, {'t': 1})
        self.assertAlmostEqual(evaluate(deriv_integral), 1, places=1)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 1, places=1)

    def test_both_boundaries_moving(self):
        # deriv(\int_{x=y=0}^{z=1} 1)
        # = \int 0 + dz - dy
        one = TegConstant(1)
        x, y, z = TegVariable('x'), TegVariable('y', 0), TegVariable('z', 1)
        integral = TegIntegral(y, z, one, x)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        check_nested_lists(self, evaluate(deriv_integral), [1, -1])

        deriv_integral = TegFwdDeriv(integral, {'y': 1, 'z': 0})
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -1)

        deriv_integral = TegFwdDeriv(integral, {'y': 0, 'z': 1})
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 1)

    def test_nested_integral_boundaries_moving(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = TegConstant(0), TegConstant(1)
        x, y, z = TegVariable('x'), TegVariable('y'), TegVariable('z', 1)

        body = TegIntegral(y, z, y * z, x)
        integral = TegIntegral(zero, one, body, y)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 2/3, places=3)

        deriv_integral = TegFwdDeriv(integral, {'z': 1})
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 2/3, places=3)

    def test_nested_integral_boundaries_moving_scaled(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = TegConstant(0), TegConstant(1)
        x, y, z = TegVariable('x'), TegVariable('y'), TegVariable('z', 1)

        body = TegIntegral(y, z, y * z, x)
        integral = TegIntegral(zero, one, y * body, y)

        deriv_integral = TegReverseDeriv(integral, TegTuple(TegConstant(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 5/12, places=3)

        deriv_integral = TegFwdDeriv(integral, {'z': 1})
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 5/12, places=3)


if __name__ == '__main__':
    unittest.main()
