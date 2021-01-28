import unittest
from unittest import TestCase
import numpy as np
import time

from teg import (
    ITeg,
    Const,
    Var,
    TegVar,
    IfElse,
    Teg,
    Tup,
    LetIn,
)

from teg.lang.extended import (
    ITegExtended,
    Delta,
    BiMap
)

from teg.math import (
    Sqr, Sqrt,
    Cos, Sin,
    ATan2, ASin
)

from teg.derivs import FwdDeriv, RevDeriv
from teg.derivs.fwd_deriv import fwd_deriv
from teg.derivs.reverse_deriv import reverse_deriv
from teg.passes.simplify import simplify
from teg.passes.reduce import reduce_to_base
from teg.passes.delta import split_expr, split_exprs, split_instance
from teg.eval import evaluate

from teg.maps.polar import polar_2d_map
from teg.maps.transform import scale, translate
from teg.maps.smoothstep import smoothstep

"""
def evaluate_c(expr: ITeg, num_samples=5000, ignore_cache=False, silent=True):
    pcount_before = time.perf_counter()
    c_code = emit(expr, target='C', num_samples=num_samples)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'Teg-to-C emit time: {pcount_after - pcount_before:.3f}s')

    pcount_before = time.perf_counter()
    binary = compileProgram(c_code)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'C compile time:     {pcount_after - pcount_before:.3f}s')

    pcount_before = time.perf_counter()
    value = runProgram(binary)
    pcount_after = time.perf_counter()
    if not silent:
        print(f'C exec time:        {pcount_after - pcount_before:.3f}s')

    return value


def evaluate(*args, **kwargs):
    fast_eval = kwargs.pop('fast_eval', True)
    if fast_eval:
        if not kwargs.get('silent', True):
            print('Evaluating in fast-mode: C')
        return evaluate_c(*args, **kwargs)
    else:
        if not kwargs.get('silent', True):
            print('Evaluating using numpy/python')
        return evaluate_numpy(*args, **kwargs)
"""


def finite_difference(expr, var, delta=0.004, num_samples=10000, silent=True, **kwargs):

    kwargs['bindings'] = dict(kwargs.get('bindings', {}))
    base = kwargs.get('bindings', {}).get(var, var.value)
    kwargs.get('bindings', {}).pop(var, None)

    assert base is not None, f'Provide a binding for {var} in order to compute it'

    # Compute upper and lower values.
    expr.bind_variable(var, base + delta)
    plus_delta = evaluate(expr, num_samples=num_samples, backend='C', **kwargs)
    if not silent:
        print('Value at base + delta', plus_delta)

    expr.bind_variable(var, base - delta)
    minus_delta = evaluate(expr, num_samples=num_samples, backend='C', **kwargs)
    if not silent:
        print('Value at base - delta', minus_delta)

    # Reset value.
    expr.bind_variable(var, base)

    gradient = (plus_delta - minus_delta) / (2 * delta)
    return gradient


def check_nested_lists(self, results, expected, places=7):
    for res, exp in zip(results, expected):
        if isinstance(res, (list, np.ndarray)):
            check_nested_lists(self, res, exp, places)
        else:
            t = (int, float, np.int64, np.float)
            err = f'Result {res} of type {type(res)} and expected {exp} of type {type(exp)}'
            assert isinstance(res, t) and isinstance(exp, t), err
            self.assertAlmostEqual(res, exp, places)


class TestArithmetic(TestCase):

    def test_linear(self):
        x = Var('x', 1)
        three_x = x + x + x
        self.assertAlmostEqual(evaluate(three_x), 3, places=3)

    def test_multiply(self):
        x = Var('x', 2)
        cube = x * x * x
        self.assertAlmostEqual(evaluate(cube), 8, places=3)

    def test_divide(self):
        x = Var('x', 2)
        y = Var('y', 4)
        fraction = x / y
        self.assertAlmostEqual(evaluate(fraction), 0.5)

        rev_res = RevDeriv(fraction, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_res), [0.25, -0.125])

        fwd_res = FwdDeriv(fraction, [(x, 1), (y, 0)])
        self.assertEqual(evaluate(fwd_res), 0.25)

        fwd_res = FwdDeriv(fraction, [(x, 0), (y, 1)])
        self.assertEqual(evaluate(fwd_res), -0.125)

        fraction = 1 / y
        self.assertAlmostEqual(evaluate(fraction), 1 / 4)

    def test_polynomial(self):
        x = Var('x', 2)
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(evaluate(poly), 14, places=3)


class TestIntegrations(TestCase):

    def test_integrate_linear(self):
        a, b = 0, 1
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x dx
        integral = Teg(a, b, x, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_integrate_sum(self):
        a, b = 0, 1
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} 2x dx
        integral = Teg(a, b, x + x, x)
        self.assertAlmostEqual(evaluate(integral), 1, places=3)

    def test_integrate_product(self):
        a, b = -1, 1
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x^2 dx
        integral = Teg(a, b, x * x, x)
        self.assertAlmostEqual(evaluate(integral), 2 / 3, places=1)

    def test_integrate_division(self):
        x = TegVar('x')
        # int_{a}^{b} x^2 dx
        integral = Teg(1, 2, 1 / x**2, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_integrate_poly(self):
        a, b = -1, 2
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x dx
        integral = Teg(a, b, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, num_samples=1000), 11.25, places=3)

    def test_integrate_poly_poly_bounds(self):
        a, b = -1, 2
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a*a}^{b} x dx
        integral = Teg(a * a + a + a, b * b + a + a, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, num_samples=1000), 11.25, places=3)


class TestNestedIntegrals(TestCase):

    def test_nested_integrals_same_variable(self):
        a, b = 0, 1
        x = TegVar('x')
        a = Var('a', a)
        b = Var('b', b)
        # \int_0^1 \int_0^1 x dx dx
        body = Teg(a, b, x, x)
        integral = Teg(a, b, body, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_nested_integrals_different_variable(self):
        a, b = 0, 1
        x = TegVar('x')
        t = TegVar('t')
        a = Var('a', a)
        b = Var('b', b)
        # \int_0^1 x * \int_0^1 t dt dx
        body = Teg(a, b, t, t)
        integral = Teg(a, b, x * body, x)
        self.assertAlmostEqual(evaluate(integral), 0.25, places=3)

    def test_integral_with_integral_in_bounds(self):
        a, b = 0, 1
        x = TegVar('x')
        t = TegVar('t')
        a = Var('a', a)
        b = Var('b', b)
        # \int_{\int_0^1 2x dx}^{\int_0^1 xdx + \int_0^1 tdt} x dx
        integral1 = Teg(a, b, x + x, x)
        integral2 = Teg(a, b, t, t) + Teg(a, b, x, x)

        integral = Teg(integral1, integral2, x, x)
        self.assertAlmostEqual(evaluate(integral), 0, places=3)


class TestConditionals(TestCase):

    def test_basic_branching(self):
        a = Var('a', 0)
        b = Var('b', 1)

        x = Var('x')
        cond = IfElse(x < 0, a, b)

        # if(x < c) 0 else 1 at x=-1
        # cond.bind_variable(x, -1)
        self.assertEqual(evaluate(cond, {x: -1}), 0)

        # if(x < 0) 0 else 1 at x=1
        # cond.bind_variable(x, 1)
        self.assertEqual(evaluate(cond, {x: 1}), 1)

    def test_integrate_branch(self):
        a = Var('a', -1)
        b = Var('b', 1)

        x = TegVar('x')
        d = Var('d', 0)
        # if(x < c) 0 else 1
        body = IfElse(x < 0, d, b)
        integral = Teg(a, b, body, x)

        # int_{a=-1}^{b=1} (if(x < c) 0 else 1) dx
        self.assertAlmostEqual(evaluate(integral), 1, places=3)

    def test_branch_in_cond(self):
        a = Var('a', 0)
        b = Var('b', 1)

        x = Var('x', -1)
        t = TegVar('t')
        # if(x < c) 0 else 1
        upper = IfElse(x < 0, a, b)
        integral = Teg(a, upper, t, t)

        # int_{a=0}^{if(x < c) 0 else 1} t dt
        self.assertEqual(evaluate(integral), 0)


class TestTups(TestCase):

    def test_tuple_basics(self):
        t = Tup(*[Const(i) for i in range(5, 15)])
        two_t = t + t
        self.assertEqual(evaluate(two_t)[0], 10)

        t_squared = t * t
        self.assertEqual(evaluate(t_squared)[0], 25)

        x = Var("x", 2)
        v1 = Tup(*[Const(i) for i in range(5, 15)])
        v2 = Tup(*[IfElse(x < 0, Const(1), Const(2)) for i in range(5, 15)])

        t_squared = v1 * v2
        self.assertEqual(evaluate(t_squared)[0], 10)

    def test_tuple_branch(self):
        x = Var("x", 2)
        v = Tup(*[Const(i) for i in range(5, 15)])
        cond = IfElse(x < 0, v, Const(1))
        res = Const(1) + cond + v
        self.assertEqual(evaluate(res)[0], 7)

        x.bind_variable(x, -1)
        self.assertEqual(evaluate(res)[0], 11)

    def test_tuple_integral(self):
        x = TegVar('x')
        v = Tup(*[Const(i) for i in range(3)])
        res = Teg(Const(0), Const(1), v * x, x)
        res, expected = evaluate(res), [0, .5, 1]
        check_nested_lists(self, res, expected)


class TestLetIn(TestCase):

    def test_let_in(self):
        x = Var('x', 2)
        y = Var('y', -1)
        f = x * y
        rev_res = RevDeriv(f, Tup(Const(1)))
        # reverse_deriv(mul(x=2, y=-1), 1, 1)
        # rev_res.deriv_expr is: let ['dx=add(0, mul(1, y=-1))', 'dy=add(0, mul(1, x=2))'] in rev_deriv0 = dx, dy
        expected = [-1, 2]

        # y = -1
        fwd_res = FwdDeriv(f, [(x, 1), (y, 0)])
        self.assertEqual(evaluate(fwd_res), -1)

        # df/dx * [df/dx, df/dy]
        # -1 * [-1, 2] = [1, -2]
        expected = [1, -2]
        res = fwd_res * rev_res
        check_nested_lists(self, evaluate(res), expected)


class TestDerivUtilities(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_rev_deriv_order(self):
        # \int_{x = [0, 1]} (x - 2 * t1 + 3 * t2 ? 0 : 1)
        x, t1, t2, t3 = TegVar('x'), Var('t1', 3/4), Var('t2', 1/3), Var('t3', 0)
        cond = IfElse(x - Const(2) * t1 + Const(3) * t2 + Const(-10) * t3 < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        # No ordering specified. Must be ordered by UID
        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [-2, 3, -10])

        # Specify normal order
        deriv_integral = RevDeriv(integral, Tup(Const(1)), output_list=[t1, t2, t3])
        check_nested_lists(self, evaluate(deriv_integral), [-2, 3, -10])

        # Specify random order
        deriv_integral = RevDeriv(integral, Tup(Const(1)), output_list=[t2, t3, t1])
        check_nested_lists(self, evaluate(deriv_integral), [3, -10, -2])

        # Specify subset
        deriv_integral = RevDeriv(integral, Tup(Const(1)), output_list=[t2, t3])
        check_nested_lists(self, evaluate(deriv_integral), [3, -10])

        # Specify singleton
        deriv_integral = RevDeriv(integral, Tup(Const(1)), output_list=[t2])
        self.assertEqual(evaluate(deriv_integral), 3)


class ActualIntegrationTest(TestCase):

    def test_deriv_inside_integral(self):
        x = TegVar('x')

        integral = Teg(Const(0), Const(1), FwdDeriv(x * x, [(x, 1)]), x)
        self.assertAlmostEqual(evaluate(integral), 1)

        integral = Teg(Const(0), Const(1), RevDeriv(x * x, Tup(Const(1))), x)
        self.assertAlmostEqual(evaluate(integral), 1)

    def test_deriv_outside_integral(self):
        x = TegVar('x')
        integral = Teg(Const(0), Const(1), x, x)

        d_integral = FwdDeriv(integral, [(x, 1)])
        self.assertEqual(evaluate(d_integral), 0)

    def test_deriv_expr(self):
        x, y = Var('x', 1), Var('y', 2)
        expr = 2 * x + y

        d_expr = FwdDeriv(expr, [(x, 1)])
        self.assertEqual(evaluate(d_expr), 2)

        d_expr = FwdDeriv(expr, [(y, 1)])
        self.assertEqual(evaluate(d_expr), 1)


class VariableBranchConditionsTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_deriv_heaviside(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        heaviside = IfElse(x < t, self.zero, self.one)
        integral = Teg(self.zero, self.one, heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), -1, places=3)

        neg_heaviside = IfElse(x < t, self.one, self.zero)
        integral = Teg(self.zero, self.one, neg_heaviside, x)

        # d/dt int_{x=0}^{1} (x<t) ? 1 : 0
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), 1, places=3)

    def test_deriv_heaviside_discontinuity_out_of_domain(self):
        x = TegVar('x')
        t = Var('t', 2)
        heaviside = IfElse(x < t, self.zero, self.one)
        integral = Teg(self.zero, self.one, heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), 0, places=3)

        t.bind_variable(t, -1)
        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), 0, places=3)

    def test_deriv_scaled_heaviside(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        body = (x + t) * heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 (x + t) * ((x<t) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        # print(simplify(fwd_dintegral))
        self.assertAlmostEqual(evaluate(fwd_dintegral), -0.5, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), -0.5, places=3)

    def test_deriv_add_heaviside(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        body = x + t + heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 x + t + ((x<t) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), 0, places=3)

    def test_deriv_sum_heavisides(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        flipped_heavyside_t = IfElse(x < t, Const(1), Const(0))
        body = flipped_heavyside_t + Const(2) * heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 1 : 0) + 2 * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1, places=3)
        return

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        # print('Simplified..')
        # print(simplify(rev_dintegral.deriv_expr))
        self.assertAlmostEqual(evaluate(rev_dintegral), -1, places=3)

        t1 = Var('t1', 0.3)
        heavyside_t1 = IfElse(x < t1, Const(0), Const(1))
        body = heavyside_t + heavyside_t1
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) + ((x<t1=0.3) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -2, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral), [-1, -1], places=3)

        body = heavyside_t + heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) + ((x<t=0.3) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -2, places=3)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), -2, places=3)

    def test_deriv_product_heavisides(self):
        x = TegVar('x')
        t = Var('t', 0.5)

        heavyside_t = IfElse(x < t, Const(0), Const(1))

        t1 = Var('t1', 0.3)
        heavyside_t1 = IfElse(x < t1, Const(0), Const(1))
        body = heavyside_t * heavyside_t1
        integral = Teg(Const(0), Const(1), body, x)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral), [-1, 0], places=3)

    def test_tzu_mao(self):
        # \int_{x=0}^1 x < t ?
        #   \int_{y=0}^1 x < t1 ?
        #     x * y : x * y^2 :
        #   \int_{y=0}^1 y < t2 ?
        #     x^2 * y : x^2 * y^2
        zero, one = Const(0), Const(1)
        x, y = TegVar('x'), TegVar('y')
        t, t1, t2 = Var('t', 0.5), Var('t1', 0.25), Var('t2', 0.75)
        if_body = Teg(zero, one, IfElse(x < t1, x * y, x * y * y), y)
        else_body = Teg(zero, one, IfElse(y < t2, x * x * y, x * x * y * y), y)
        integral = Teg(zero, one, IfElse(x < t, if_body, else_body), x)

        expected = [0.048, 0.041, 0.054]
        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral), expected, places=2)

        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 0), (t2, 0)])
        dt = evaluate(fwd_dintegral, num_samples=1000)

        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1), (t2, 0)])
        dt1 = evaluate(fwd_dintegral, num_samples=1000)

        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 0), (t2, 1)])
        dt2 = evaluate(fwd_dintegral, num_samples=1000)

        check_nested_lists(self, [dt, dt1, dt2], expected, places=2)

    def test_single_integral_example(self):
        # deriv(\int_{x=0}^1 (x < theta ? 1 : x * theta))
        zero, one = Const(0), Const(1)
        x, theta = TegVar('x'), Var('theta', 0.5)

        body = IfElse(x < theta, one, x * theta)
        integral = Teg(zero, one, body, x)
        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 1.125, places=2)

    def test_double_integral(self):
        # \int_{x=0}^1
        #   deriv(
        #       \int_{y=0}^1
        #           (x < y ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, y = TegVar('x'), TegVar('y')

        body = IfElse(x < y, zero, one)
        integral = Teg(zero, one, body, x)
        body = RevDeriv(integral, Tup(Const(1)))
        # print((body.deriv_expr))
        double_integral = Teg(zero, one, body, y)
        self.assertAlmostEqual(evaluate(double_integral, num_samples=100), -1, places=3)

    def test_nested_integral_moving_discontinuity(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, y, t = TegVar('x'), TegVar('y'), Var('t', 0.5)

        body = Teg(zero, one, IfElse(x < t, zero, one), x)
        integral = Teg(zero, one, y * body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        # print(simplify(deriv_integral.deriv_expr))
        self.assertAlmostEqual(evaluate(deriv_integral), -0.5, places=3)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), -0.5, places=3)

    def test_nested_discontinuity_integral(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, t1, t = TegVar('x'), Var('t1', 0.5), Var('t', 0.4)

        body = IfElse(t < t1, IfElse(x < t, zero, one), one)
        integral = Teg(zero, one, body, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=3)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=3)

    def test_nested_discontinuities(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, t1, t = TegVar('x'), Var('t1', 0.5), Var('t', 0.4)

        x, y = TegVar('x'), TegVar('y')
        t, t1, t2 = Var('t', 0.5), Var('t1', 0.25), Var('t2', 0.75)
        if_body = Teg(zero, one, IfElse(t1 < y, 1, 2), y)
        else_body = Teg(zero, one, IfElse(t2 < y, 3, 4), y)
        integral = Teg(zero, one, IfElse(t < x, if_body, else_body), x)

        d_t1 = finite_difference(integral, t1, delta=0.01, num_samples=1000)
        d_t2 = finite_difference(integral, t2, delta=0.01, num_samples=1000)
        d_t = finite_difference(integral, t, delta=0.01, num_samples=1000)

        deriv_integral = RevDeriv(integral, Tup(Const(1)), output_list=[t, t1, t2])
        check_nested_lists(self, evaluate(simplify(deriv_integral)),
                           [d_t, d_t1, d_t2], places=2)

        fwd_dt = evaluate(simplify(FwdDeriv(integral, [(t, 1)])))
        fwd_dt1 = evaluate(simplify(FwdDeriv(integral, [(t1, 1)])))
        fwd_dt2 = evaluate(simplify(FwdDeriv(integral, [(t2, 1)])))

        check_nested_lists(self, [fwd_dt, fwd_dt1, fwd_dt2],
                           [d_t, d_t1, d_t2], places=2)


class MovingBoundaryTest(TestCase):

    def test_moving_upper_boundary(self):
        # deriv(\int_{x=0}^{t=1} xt)
        # 0.5 + 1 - 0 = 1.5
        zero = Const(0)
        x, t = TegVar('x'), Var('t', 1)
        integral = Teg(zero, t, x * t, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 1.5)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 1.5)

    def test_moving_lower_boundary(self):
        # deriv(\int_{x=t=-1}^{1} xt)
        # 0 + 0 + 1 = 1
        one = Const(1)
        x, t = TegVar('x'), Var('t', -1)
        integral = Teg(t, one, x * t, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=1)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=1)

    def test_both_boundaries_moving(self):
        # deriv(\int_{x=y=0}^{z=1} 1)
        # = \int 0 + dz - dy
        one = Const(1)
        x, y, z = TegVar('x'), Var('y', 0), Var('z', 1)
        integral = Teg(y, z, one, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [-1, 1])

        deriv_integral = FwdDeriv(integral, [(y, 1), (z, 0)])
        self.assertAlmostEqual(evaluate(deriv_integral), -1)

        deriv_integral = FwdDeriv(integral, [(y, 0), (z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 1)

    def test_nested_integral_boundaries_moving(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = Const(0), Const(1)
        x, y, z = TegVar('x'), TegVar('y'), Var('z', 1)

        body = Teg(y, z, y * z, x)
        integral = Teg(zero, one, body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 2/3, places=3)

        deriv_integral = FwdDeriv(integral, [(z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 2/3, places=3)

    def test_nested_integral_boundaries_moving_scaled(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = Const(0), Const(1)
        x, y, z = TegVar('x'), TegVar('y'), Var('z', 1)

        body = Teg(y, z, y * z, x)
        integral = Teg(zero, one, y * body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 5/12, places=3)

        deriv_integral = FwdDeriv(integral, [(z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 5/12, places=3)

    def test_affine_moving_boundary(self):
        # deriv(\int_{x=a - b}^{a + b} 1)
        x, a, b = TegVar('x'), Var('a', 0), Var('b', 1)

        integral = Teg(a - b, a + b, 1, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [0, 2], places=3)

        deriv_integral = FwdDeriv(integral, [(a, 1), (b, 0)])
        # print(deriv_integral.deriv_expr)
        self.assertAlmostEqual(evaluate(deriv_integral), 0, places=3)

        deriv_integral = FwdDeriv(integral, [(a, 0), (b, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 2, places=3)

    def test_affine_moving_boundary_variable_body(self):
        # deriv(\int_{x=a - b}^{a + b} x)
        x, a, b = TegVar('x'), Var('a', 0), Var('b', 1)
        integral = Teg(a - b, a + b, x, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [2, 0], places=3)

        deriv_integral = FwdDeriv(integral, [(a, 1), (b, 0)])
        self.assertAlmostEqual(evaluate(deriv_integral), 2, places=3)

        deriv_integral = FwdDeriv(integral, [(a, 0), (b, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 0, places=3)


class PiecewiseAffineTest(TestCase):

    def test_and_different_locations(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        t1 = Var('t1', 0.3)
        heavyside_t = IfElse((x < t) & (x < t1), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral), [0, -1])

    def test_or_different_locations(self):
        x = TegVar('x')
        t = Var('t', 0.5)
        t1 = Var('t1', 0.3)
        heavyside_t = IfElse((x < t) | (x < t1), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 0)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral), [-1, 0])

    def test_compound_exprs(self):
        x = TegVar('x')
        t = Var('t', 0.8)
        t1 = Var('t1', 0.3)
        t2 = Var('t2', 0.1)
        if_else = IfElse(((x < t) & (x < t1 + t1)) & (x < t2 + 1), Const(0), Const(1))

        integral = Teg(0, 1, if_else, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1), (t2, 0)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -2, places=3)


class AffineConditionsTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_affine_condition_simple(self):
        x, t = TegVar('x'), Var('t', 0.25)
        cond = IfElse(x < Const(2) * t, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        # print("Simplified: ", simplify(deriv_integral.deriv_expr))
        self.assertAlmostEqual(evaluate(deriv_integral.deriv_expr), -2)

    def test_affine_condition_with_constants(self):
        x, t = TegVar('x'), Var('t', 7/4)
        cond = IfElse(x + Const(2) * t - Const(4) < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 2)

    def test_affine_condition_2x(self):
        x, t = TegVar('x'), Var('t', 7/4)
        cond = IfElse(Const(2) * x + Const(2) * t - Const(4) < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 1)

    def test_scaled_affine_condition(self):
        x, t = TegVar('x'), Var('t', 0.5)
        cond = IfElse(x - t < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, 2 * cond, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), -2)

    def test_affine_condition_multivariable(self):
        # \int_{x = [0, 1]} (x - 2 * t1 + 3 * t2 ? 0 : 1)
        x, t1, t2 = TegVar('x'), Var('t1', 3/4), Var('t2', 1/3)
        cond = IfElse(x - Const(2) * t1 + Const(3) * t2 < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [-2, 3])

        deriv_integral = FwdDeriv(integral, [(t1, 1), (t2, 0)])
        self.assertEqual(evaluate(deriv_integral), -2)

        deriv_integral = FwdDeriv(integral, [(t1, 0), (t2, 1)])
        self.assertEqual(evaluate(deriv_integral), 3)

    def test_affine_condition_multivariable_multiintegral(self):
        x, y = TegVar('x'), TegVar('y')
        t1, t2 = Var('t1', 0.25), Var('t2', 1)
        cond = IfElse((x - y) + (Const(2) * t1 + Const(3) * t2 + Const(-3)) < 0, self.one, self.zero)
        body = Teg(self.zero, self.one, cond, y)
        integral = Teg(self.zero, self.one, body, x)

        d_t1 = -1.0
        d_t2 = -1.5

        deriv_integral = RevDeriv(integral, Tup(Const(1)))

        check_nested_lists(self,
                           evaluate(simplify(deriv_integral), num_samples=5000),
                           [d_t1, d_t2], places=2)

        deriv_integral = FwdDeriv(integral, [(t1, 1), (t2, 0)])
        sd = simplify(deriv_integral.deriv_expr)


        self.assertAlmostEqual(evaluate(sd, num_samples=5000),
                               d_t1, places=2)

        deriv_integral = FwdDeriv(integral, [(t1, 0), (t2, 1)])
        self.assertAlmostEqual(evaluate(simplify(deriv_integral), num_samples=5000),
                               d_t2, places=2)

    def test_affine_condition_multivariable_multiintegral_parametric(self):
        x, y = TegVar('x'), TegVar('y')
        t1, t2 = Var('t1', 1), Var('t2', 0)
        cond = IfElse((t2 * x - t1 * y) + 0.5 < 0, self.one, self.zero)
        body = Teg(self.zero, self.one, cond, y)
        integral = Teg(self.zero, self.one, body, x)

        d_t1 = finite_difference(integral, t1, delta=0.01, num_samples=10000)
        d_t2 = finite_difference(integral, t2, delta=0.01, num_samples=10000)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        sd = simplify(deriv_integral)

        check_nested_lists(self, evaluate(sd, num_samples=5000),
                           [d_t1, d_t2], places=2)

    def test_multi_affine_condition_multivariable_multiintegral_parametric(self):
        x, y = TegVar('x'), TegVar('y')
        t1, t2 = Var('t1', 1), Var('t2', 1)
        cond1 = IfElse((t2 * x - t1 * y) < 0, self.one, self.zero)
        # cond1 = IfElse((x - y) + (Const(0.5) * t1 - Const(0.5) * t2) < 0, self.one, self.zero)
        t3, t4 = Var('t3', 1), Var('t4', 1)
        cond2 = IfElse((t3 * x + t4 * y) - 1 < 0, self.one, self.zero)
        # cond2 = IfElse((x + y) + (Const(0.5) * t3 - Const(0.5) * t4) < 0, self.one, self.zero)

        body = Teg(self.zero, self.one, cond1 * cond2, y)
        integral = Teg(self.zero, self.one, body, x)

        d_t1 = 0.125  # finite_difference(integral, t1, delta=0.005, num_samples=10000)
        d_t2 = -0.125  # finite_difference(integral, t2, delta=0.005, num_samples=10000)
        d_t3 = -0.125  # finite_difference(integral, t3, delta=0.005, num_samples=10000)
        d_t4 = -0.3725  # finite_difference(integral, t4, delta=0.005, num_samples=10000)

       
        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(simplify(deriv_integral), num_samples=5000, backend='C'),
                           [d_t1, d_t2, d_t3, d_t4], places=2)


def compare_eval_methods(self, *args, **kwargs):
    places = kwargs.pop("places", 7)
    numpy_output = evaluate(*args, **kwargs, backend='numpy')
    c_output = evaluate(*args, **kwargs, backend='C')
    if type(c_output) is list:
        assert len(numpy_output) == len(c_output), "Output lengths do not match"
        check_nested_lists(self, numpy_output, c_output, places=places)
    else:
        self.assertAlmostEqual(numpy_output, c_output, places=places)


def deep_compare_eval_methods(self, expr, depth=0, **kwargs):
    """
        Utility that compares various codegen methods of
        various subtrees upto a specific depth.
        Useful to pinpoint codegen problems in large programs.
        Raises ValueError when it encounters the lowest subtree that produces inconsistent results
    """
    if isinstance(expr, Teg):
        deep_compare_eval_methods(self, expr.lower, depth=depth+1, **kwargs)
        deep_compare_eval_methods(self, expr.upper, depth=depth+1, **kwargs)
        for val in [0, 0.2, 0.4, 0.8, 1]:
            print(f'{expr.dvar}:{val}')
            expr.body.bind_variable(expr.dvar, value=val)
            deep_compare_eval_methods(self, expr.body, depth=depth+1, **kwargs)

    else:
        if hasattr(expr, 'children'):
            [deep_compare_eval_methods(self, child, depth=depth+1, **kwargs) for child in expr.children]

    if isinstance(expr, Var):
        if expr.value is None:
            expr.value = 0.5
        return

    if depth > 9:
        return

    print(f'Node at depth {depth}')
    try:
        compare_eval_methods(self, expr, **kwargs)
    except AssertionError as e:
        print(e)
        print('SUB-EXPR: ')
        print(expr)
        raise ValueError


class FastEvalTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_let_inner(self):
        x = TegVar('x')
        t = Var('t', 1)
        integral = Teg(self.zero, self.one, LetIn([t], [Const(1)], x*t), x)
        compare_eval_methods(self, integral, places=2)

    def test_let_outer(self):
        x = TegVar('x')
        t = Var('t', 1)
        integral = LetIn([t], [Const(1)], t * Teg(self.zero, self.one, x * t, x))
        compare_eval_methods(self, integral, places=2)

    def test_overlapping_let(self):
        x = TegVar('x')
        t = Var('t', 1)
        integral = LetIn([t], [Const(1)], t * Teg(self.zero, self.one, LetIn([t], [Const(1)], x*t), x))
        compare_eval_methods(self, integral, places=2)

    def test_tuple_integration(self):
        x = TegVar('x')
        t = Var('t', 1)
        integral = LetIn([t], [Const(1)], t * Teg(self.zero, self.one, LetIn([t], [Const(1)], Tup(x*t, 2*x*t)), x))
        compare_eval_methods(self, integral, places=2)


class MathFunctionsTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_sqr(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body = Sqr(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral = Teg(self.zero, self.one, Teg(self.zero, Sqrt(x_val), body, x), y)
            self.assertAlmostEqual(evaluate(integral, num_samples=100),
                                   (np.sqrt(x_val) ** 3 / 3), places=2)

    def test_sqr_backends(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body = Sqr(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral = Teg(self.zero, self.one, Teg(self.zero, x_val, body, x), y)
            compare_eval_methods(self, integral, places=2)

    def test_sqrt(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body = Sqrt(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral = Teg(self.zero, self.one, Teg(self.zero, x_val, body, x), y)
            self.assertAlmostEqual(evaluate(integral, num_samples=100),
                                   (x_val ** (1.5) / 1.5), places=2)

    def test_sqrt_backends(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body = Sqrt(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral = Teg(self.zero, self.one, Teg(self.zero, x_val, body, x), y)
            compare_eval_methods(self, integral, places=2)

    def test_trigonometry(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body_cos = Cos(x - theta)
        body_sin = Sin(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral_sin = Teg(self.zero, self.one, Teg(self.zero, x_val, body_sin, x), y)
            integral_cos = Teg(self.zero, self.one, Teg(self.zero, x_val, body_cos, x), y)
            self.assertAlmostEqual(evaluate(integral_sin, num_samples=400),
                                   (1-np.cos(x_val)), places=2)
            self.assertAlmostEqual(evaluate(integral_cos, num_samples=400),
                                   (np.sin(x_val)), places=2)

    def test_trigonometry_backends(self):
        x, y = TegVar('x'), TegVar('y')
        theta = Var('theta', 0)
        body_cos = Cos(x - theta)
        body_sin = Sin(x - theta)

        for x_val in [0, 0.5, 1, 1.5]:
            integral_sin = Teg(self.zero, self.one, Teg(self.zero, x_val, body_sin, x), y)
            integral_cos = Teg(self.zero, self.one, Teg(self.zero, x_val, body_cos, x), y)
            compare_eval_methods(self, integral_sin, places=2)
            compare_eval_methods(self, integral_cos, places=2)

    def test_arcs(self):
        x = Var('x')
        y = Var('y')
        body_atan2 = ATan2(Tup(x, y))

        self.assertAlmostEqual(evaluate(body_atan2, {x: 0, y: -1}), np.pi, places=4)
        self.assertAlmostEqual(evaluate(body_atan2, {x: 0, y: 1}), 0, places=4)
        self.assertAlmostEqual(evaluate(body_atan2, {x: 1, y: 0}), np.pi/2, places=4)
        self.assertAlmostEqual(evaluate(body_atan2, {x: -1, y: 0}), -np.pi/2, places=4)
        compare_eval_methods(self, body_atan2, {x: 0, y: -1}, places=2)
        compare_eval_methods(self, body_atan2, {x: 0, y: 1}, places=2)
        compare_eval_methods(self, body_atan2, {x: 1, y: 0}, places=2)
        compare_eval_methods(self, body_atan2, {x: -1, y: 0}, places=2)

        body_asin = ASin(x)

        self.assertAlmostEqual(evaluate(body_asin, {x: -1}), -np.pi/2, places=4)
        self.assertAlmostEqual(evaluate(body_asin, {x: 0}), 0, places=4)
        self.assertAlmostEqual(evaluate(body_asin, {x: 1}), np.pi/2, places=4)
        compare_eval_methods(self, body_asin, {x: 1}, places=2)
        compare_eval_methods(self, body_asin, {x: 0}, places=2)
        compare_eval_methods(self, body_asin, {x: -1}, places=2)


class DeltaTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_delta_normal(self):
        x = TegVar('x')
        a, b = Var('a'), Var('b')
        integral = Teg(a, b, Delta(x), x)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1}), 1)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: 0.1, b: 1}), 0)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: -0.1}), 0)

    def test_split_simple(self):
        x = TegVar('x')
        k = Var('k', 1)
        expr = Delta(x)
        integrand = expr * 2 + x + k
        split_direct = Teg(-1, 1, split_instance(expr, integrand), x)
        split_indirect = Teg(-1, 1, split_expr(Delta(x), integrand), x)
        split_multiple = Teg(-1, 1, split_exprs([expr, k], integrand), x)
        split_invalid = split_instance(Delta(x), integrand)

        self.assertAlmostEqual(evaluate(reduce_to_base(split_direct)), 2)
        self.assertAlmostEqual(evaluate(reduce_to_base(split_indirect)), 2)
        self.assertAlmostEqual(evaluate(reduce_to_base(split_multiple)), 4)
        self.assertIsNone(split_invalid)

    def test_split_through_let(self):
        x = TegVar('x')
        x_ = Var('x_')
        y_ = Var('y_')

        k = Var('k', 2)
        expr = Delta(x)

        one_let = LetIn([x_], [2 * expr], x_ + x + k)
        two_let = LetIn([x_, y_], [2 * expr, expr * k], x_ + x + y_ * k)
        expr1 = Delta(x)
        expr2 = Delta(x)
        two_let_linear_k = LetIn([x_, y_], [k * expr1, expr2 * 2], x_ + x + y_ * k)
        nested_let = LetIn([x_], [2 * expr], LetIn([y_], [3 * expr + x_], x_ + k * y_ + expr))

        # Instance is duplicated.
        self.assertRaises(AssertionError, lambda: split_instance(expr, two_let))
        self.assertRaises(AssertionError, lambda: split_instance(expr, nested_let))

        self.assertAlmostEqual(evaluate(reduce_to_base(Teg(-1, 1, split_instance(expr, one_let), x))), 2, places=3)
        self.assertAlmostEqual(evaluate(reduce_to_base(Teg(-1, 1, split_expr(expr, two_let), x))), 6, places=3)
        self.assertAlmostEqual(evaluate(reduce_to_base(Teg(-1, 1, split_expr(k, two_let_linear_k), x))), 6, places=3)
        self.assertAlmostEqual(evaluate(reduce_to_base(Teg(-1, 1, split_expr(expr, nested_let), x))), 13, places=3)

    def test_manual_bimap(self):
        x = TegVar('x')
        x_ = TegVar('x_')
        a, b = Var('a'), Var('b')
        t = Var('t')
        integral = Teg(a, b, BiMap(x_,
                                   [x_], [x - t],
                                   [x], [x_ + t],
                                   inv_jacobian=Const(1),
                                   target_lower_bounds=[x.lower_bound() - t],
                                   target_upper_bounds=[x.upper_bound() - t]), x)

        t_val, a_val, b_val = 0, -1, 1
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: a_val, b: b_val, t: t_val}),
                               (b_val**2 - a_val**2) / 2 - t_val * (b_val - a_val), places=3)
        t_val, a_val, b_val = 2, -4, 2
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: a_val, b: b_val, t: t_val}),
                               (b_val**2 - a_val**2) / 2 - t_val * (b_val - a_val), places=3)

    def test_manual_bimap_with_delta(self):
        x = TegVar('x')
        x_ = TegVar('x_')
        a, b = Var('a'), Var('b')
        t = Var('t')
        integral = Teg(a, b, BiMap(Delta(x_), [x_], [x - t], [x], [x_ + t], inv_jacobian=Const(1),
                                   target_lower_bounds=[x.lower_bound() - t], target_upper_bounds=[x.upper_bound() - t]), x)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 0}), 1)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 2}), 0)

    def test_delta_single_axis(self):
        x = TegVar('x')
        a, b = Var('a'), Var('b')
        t = Var('t')
        integral = Teg(a, b, Delta(x - t), x)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 0}), 1)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 2}), 0)

    def test_delta_scaled(self):
        x = TegVar('x')
        a, b = Var('a'), Var('b')
        t = Var('t')
        integral = Teg(a, b, Delta(x - t) + 2 * Delta(x - t), x)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 0}), 3)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral), {a: -1, b: 1, t: 2}), 0)

    def test_delta_1d_affine(self):
        x = TegVar('x')
        a = Var('a')

        integral = Teg(0, 1, Delta(2 * x - a), x)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 1}), 0.5)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 3}), 0)

        integral = Teg(0, 1, Delta(a * x - 2), x)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 4}), 0.25)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 1}), 0)

    def test_delta_2d_single_axis(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        integral = Teg(0, 1, Teg(0, 1, Delta(x - t), x), y)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {t: 0.5}), 1)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {t: 2}), 0)

    def test_delta_2d_affine(self):
        x = TegVar('x')
        y = TegVar('y')
        a = Var('a')
        b = Var('b')

        integral = Teg(0, 1, Teg(0, 1, Delta(a * x - b * y), x), y)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 1, b: 1}), 1, places=3)
        self.assertAlmostEqual(evaluate(simplify(reduce_to_base(integral)), {a: 0, b: 2}), 0, places=3)

    def test_delta_product(self):
        x = TegVar('x')
        y = TegVar('y')

        a_x = Var('a_x')
        a_y = Var('a_y')

        b_x = Var('b_x')
        b_y = Var('b_y')

        integral = Teg(a_y, b_y, Teg(a_x, b_x, Delta(x) * Delta(y), x), y)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral),
                                        {a_x: -1, b_x: 1, a_y: -1, b_y: 1}), 1, places=3)
        self.assertAlmostEqual(evaluate(reduce_to_base(integral),
                                        {a_x: 1, b_x: 2, a_y: 1, b_y: 2}), 0, places=3)


class SecondDerivativeTests(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_second_deriv_simple(self):
        x = Var('x', 1)
        y = Var('y', 1)
        expr = x * x * y * y * y
        dx2 = fwd_deriv(fwd_deriv(expr, [(x, 1)]), [(x, 1)])
        dy2 = fwd_deriv(fwd_deriv(expr, [(y, 1)]), [(y, 1)])
        dyx = fwd_deriv(fwd_deriv(expr, [(y, 1)]), [(x, 1)])
        dxy = fwd_deriv(fwd_deriv(expr, [(x, 1)]), [(y, 1)])

        x_val = 2
        y_val = 1
        self.assertAlmostEqual(evaluate(dx2, bindings={x: x_val, y: y_val}), 2 * (y_val ** 3), places=2)
        self.assertAlmostEqual(evaluate(dy2, bindings={x: x_val, y: y_val}), 6 * (x_val ** 2) * (y_val), places=2)
        self.assertAlmostEqual(evaluate(dxy, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)
        self.assertAlmostEqual(evaluate(dyx, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)

        _, (dx, dy) = reverse_deriv(expr, output_list=[x, y])
        _, (dx2, dxy) = reverse_deriv(dx, output_list=[x, y])
        _, (dyx, dy2) = reverse_deriv(dy, output_list=[x, y])

        self.assertAlmostEqual(evaluate(dx2, bindings={x: x_val, y: y_val}), 2 * (y_val ** 3), places=2)
        self.assertAlmostEqual(evaluate(dy2, bindings={x: x_val, y: y_val}), 6 * (x_val ** 2) * (y_val), places=2)
        self.assertAlmostEqual(evaluate(dxy, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)
        self.assertAlmostEqual(evaluate(dyx, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)

    def test_second_deriv_integral(self):
        k = TegVar('k')
        x = Var('x', 1)
        y = Var('y', 1)
        expr = Teg(0, 1, 2 * x * x * y * y * y * k, k)
        dx2 = fwd_deriv(fwd_deriv(expr, [(x, 1)]), [(x, 1)])
        dy2 = fwd_deriv(fwd_deriv(expr, [(y, 1)]), [(y, 1)])
        dyx = fwd_deriv(fwd_deriv(expr, [(y, 1)]), [(x, 1)])
        dxy = fwd_deriv(fwd_deriv(expr, [(x, 1)]), [(y, 1)])

        x_val = 2
        y_val = 1
        self.assertAlmostEqual(evaluate(dx2, bindings={x: x_val, y: y_val}), 2 * (y_val ** 3), places=2)
        self.assertAlmostEqual(evaluate(dy2, bindings={x: x_val, y: y_val}), 6 * (x_val ** 2) * (y_val), places=2)
        self.assertAlmostEqual(evaluate(dxy, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)
        self.assertAlmostEqual(evaluate(dyx, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)

        _, (dx, dy) = reverse_deriv(expr, output_list=[x, y])
        _, (dx2, dxy) = reverse_deriv(dx, output_list=[x, y])
        _, (dyx, dy2) = reverse_deriv(dy, output_list=[x, y])

        self.assertAlmostEqual(evaluate(dx2, bindings={x: x_val, y: y_val}), 2 * (y_val ** 3), places=2)
        self.assertAlmostEqual(evaluate(dy2, bindings={x: x_val, y: y_val}), 6 * (x_val ** 2) * (y_val), places=2)
        self.assertAlmostEqual(evaluate(dxy, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)
        self.assertAlmostEqual(evaluate(dyx, bindings={x: x_val, y: y_val}), 6 * (x_val) * (y_val ** 2), places=2)

    def test_second_deriv_discontinuous(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        # Square region with area t^2
        expr = Teg(0, 1,
                   Teg(0, 1,
                       IfElse(x < t, 1, 0) * IfElse(y < t, 1, 0),
                       x),
                   y)

        dt = simplify(reduce_to_base(simplify(fwd_deriv(expr, [(t, 1)]))))
        dt2 = simplify(reduce_to_base(simplify(fwd_deriv(simplify(fwd_deriv(expr, [(t, 1)])), [(t, 1)]))))

        self.assertAlmostEqual(evaluate(dt, bindings={t: 0.5}, num_samples=1000), 1, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: 0.25}, num_samples=1000), 0.5, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: 1.5}, num_samples=1000), 0, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: -1}, num_samples=1000), 0, places=2)

        self.assertAlmostEqual(evaluate(dt2, bindings={t: 0.5}, num_samples=1000), 2, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: 0.25}, num_samples=1000), 2, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: 1.5}, num_samples=1000), 0, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: -1}, num_samples=1000), 0, places=2)

        _, _dt = reverse_deriv(expr, output_list=[t])
        dt = simplify(reduce_to_base(simplify(_dt)))

        _, _dt2 = reverse_deriv(simplify(_dt), output_list=[t])
        dt2 = simplify(reduce_to_base(simplify(_dt2)))

        self.assertAlmostEqual(evaluate(dt, bindings={t: 0.5}, num_samples=1000), 1, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: 0.25}, num_samples=1000), 0.5, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: 1.5}, num_samples=1000), 0, places=2)
        self.assertAlmostEqual(evaluate(dt, bindings={t: -1}, num_samples=1000), 0, places=2)

        self.assertAlmostEqual(evaluate(dt2, bindings={t: 0.5}, num_samples=1000), 2, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: 0.25}, num_samples=1000), 2, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: 1.5}, num_samples=1000), 0, places=2)
        self.assertAlmostEqual(evaluate(dt2, bindings={t: -1}, num_samples=1000), 0, places=2)


class TransformationTests(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_translation(self):
        x = TegVar('x')
        y = TegVar('y')
        t1 = Var('t1')
        t2 = Var('t2')

        translate_map, (x_, y_) = translate([x, y], [t1, t2])

        # Derivative of threshold only.
        integral = Teg(self.zero, self.one,
                       Teg(self.zero, self.one,
                           translate_map(IfElse(x_ + y_ > 1, 1, 0)), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t1, t2])
        fd_d_t1 = 1.0  # finite_difference(reduce_to_base(integral), t1, bindings={t1: 0, t2: 0})
        fd_d_t2 = 1.0  # finite_difference(reduce_to_base(integral), t2, bindings={t1: 0, t2: 0})

        d_t_expr = reduce_to_base(d_t_expr)

        check_nested_lists(self,
                           evaluate(d_t_expr, num_samples=1000, bindings={t1: 0, t2: 0}),
                           [fd_d_t1, fd_d_t2],
                           places=3)

    def test_scaling(self):
        x = TegVar('x')
        y = TegVar('y')
        t1 = Var('t1')
        t2 = Var('t2')

        scale_map, (x_, y_) = scale([x, y], [t1, t2])

        # Derivative of threshold only.
        integral = Teg(self.zero, self.one,
                       Teg(self.zero, self.one,
                           scale_map(IfElse(x_ + y_ > 1, 1, 0)), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t1, t2])
        fd_d_t1 = finite_difference(reduce_to_base(integral), t1, bindings={t1: 1, t2: 1})
        fd_d_t2 = finite_difference(reduce_to_base(integral), t2, bindings={t1: 1, t2: 1})

        d_t_expr = reduce_to_base(d_t_expr)

        check_nested_lists(self,
                           evaluate(d_t_expr, num_samples=1000, bindings={t1: 1, t2: 1}),
                           [fd_d_t1, fd_d_t2],
                           places=2)

    def test_composed_transforms(self):
        x = TegVar('x')
        y = TegVar('y')

        t1 = Var('t1')
        t2 = Var('t2')

        t3 = Var('t3')
        t4 = Var('t4')

        scale_map, (x_s, y_s) = scale([x, y], [t1, t2])
        translate_map, (x_st, y_st) = translate([x_s, y_s], [t3, t4])

        # Derivative of threshold only.
        integral = Teg(self.zero, self.one,
                       Teg(self.zero, self.one,
                           scale_map(translate_map(IfElse(x_st + y_st > 1, 1, 0))), y
                           ), x
                       )

        bindings = {t1: 1, t2: 1, t3: 0, t4: 0}
        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t1, t2, t3, t4])
        fd_d_t1 = finite_difference(reduce_to_base(integral), t1, bindings=bindings)
        fd_d_t2 = finite_difference(reduce_to_base(integral), t2, bindings=bindings)
        fd_d_t3 = finite_difference(reduce_to_base(integral), t3, bindings=bindings)
        fd_d_t4 = finite_difference(reduce_to_base(integral), t4, bindings=bindings)

        d_t_expr = reduce_to_base(d_t_expr)

        check_nested_lists(self,
                           evaluate(d_t_expr, num_samples=1000, bindings=bindings),
                           [fd_d_t1, fd_d_t2, fd_d_t3, fd_d_t4],
                           places=2)


class HyperbolicMapTests(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)
        self.near_zero = Const(10e-3)
        self.near_one = Const(1 - 10e-3)

    def test_hyperbolic_map(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        # Derivative of threshold only.
        integral = Teg(Const(10e-4), self.one,
                       Teg(Const(10e-4), self.one,
                           IfElse(x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000, bindings={t: 0.5}), fd_d_t, places=2)

        # Derivative of threshold with scaling
        integral = Teg(Const(0.3), self.one,
                       Teg(Const(0.3), self.one,
                           IfElse(2 * x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000, bindings={t: 0.5}), fd_d_t, places=2)

    def test_hyperbolic_mirrored(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        # Derivative of threshold with negative scaling
        integral = Teg(-self.one, Const(0.3),
                       Teg(Const(0.3), self.one,
                           IfElse(x * y < -t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000, bindings={t: 0.5}), fd_d_t, places=2)

        # Derivative of threshold with negative scaling
        integral = Teg(Const(0.3), self.one,
                       Teg(-self.one, Const(0.3),
                           IfElse(- x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000, bindings={t: 0.5}), fd_d_t, places=2)

    def test_transformed_hyperbolic_conditions(self):
        x, y = TegVar('x'), TegVar('y')
        t1, t2 = Var('t1'), Var('t2')
        t3, t4 = Var('t3'), Var('t4')
        t = Var('t')

        scale_map, (x_s, y_s) = scale([x, y], [t1, t2])
        translate_map, (x_st, y_st) = translate([x_s, y_s], [t3, t4])

        bindings = {t: 0.25, t1: 0.9, t2: 0.9, t3: 0.1, t4: 0.1}
        # Derivative of threshold only.
        x_lb, x_ub = Var('x_lb', 0.1), Var('x_ub', 0.9)
        y_lb, y_ub = Var('y_lb', 0.1), Var('y_ub', 0.9)
        integral = Teg(x_lb, x_ub,
                       Teg(y_lb, y_ub,
                           scale_map(translate_map(IfElse(x_st * y_st > t, 1, 0))), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t, t1, t2, t3, t4])

        fd_d_t = finite_difference(reduce_to_base(integral), t, bindings=bindings)
        fd_d_t1 = finite_difference(reduce_to_base(integral), t1, bindings=bindings)
        fd_d_t2 = finite_difference(reduce_to_base(integral), t2, bindings=bindings)
        fd_d_t3 = finite_difference(reduce_to_base(integral), t3, bindings=bindings)
        fd_d_t4 = finite_difference(reduce_to_base(integral), t4, bindings=bindings)

        d_t_expr = reduce_to_base(d_t_expr)
        check_nested_lists(self, evaluate(d_t_expr, num_samples=1000, bindings=bindings),
                           [fd_d_t, fd_d_t1, fd_d_t2, fd_d_t3, fd_d_t4], places=2)

    def test_hyperbolic_map_negative(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        # Derivative of threshold only.
        integral = Teg(-self.one, Const(-10e-4),
                       Teg(-self.one, Const(-10e-4),
                           IfElse(x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000), fd_d_t, places=2)

        # Derivative of threshold with scaling
        integral = Teg(-self.one, Const(-0.3),
                       Teg(-self.one, Const(-0.3),
                           IfElse(2 * x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000), fd_d_t, places=2)

    def test_hyperbolic_full_domain(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        c_xy = Var('c_xy')

        # Derivative of threshold only.
        integral = Teg(-self.one, self.one,
                       Teg(-self.one, self.one,
                           IfElse(x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000), fd_d_t, places=2)

        # Derivative of threshold with scaling
        integral = Teg(-self.one, self.one,
                       Teg(-self.one, self.one,
                           IfElse(c_xy * x * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        _, d_cxy_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[c_xy])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5, c_xy: 2.0})
        fd_d_cxy = finite_difference(integral, c_xy, bindings={t: 0.5, c_xy: 2.0})
        d_t_expr = reduce_to_base(d_t_expr)
        d_cxy_expr = reduce_to_base(d_cxy_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000, bindings={t: 0.5, c_xy: 2.0}), fd_d_t, places=2)
        self.assertAlmostEqual(evaluate(d_cxy_expr, num_samples=1000, bindings={t: 0.5, c_xy: 2.0}), fd_d_cxy, places=2)

    def test_non_centered_hyperbola(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        # Derivative of threshold only.
        integral = Teg(-self.one, self.one,
                       Teg(-self.one, self.one,
                           IfElse(x * y + 0.1 * x + 0.1 * y > t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000), fd_d_t, places=2)

        # Derivative of threshold (mirrored axes).
        integral = Teg(-self.one, self.one,
                       Teg(-self.one, self.one,
                           IfElse(x * y + 0.1 * x + 0.1 * y < -t, 1, 0), y
                           ), x
                       )

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t])
        fd_d_t = finite_difference(integral, t, bindings={t: 0.5})
        d_t_expr = reduce_to_base(d_t_expr)
        self.assertAlmostEqual(evaluate(d_t_expr, num_samples=1000), fd_d_t, places=2)

    def test_parametric_hyperbola(self):
        x = TegVar('x')
        y = TegVar('y')

        t = Var('t')
        c_xy = TegVar('c_xy')
        c_x = TegVar('c_x')
        c_y = TegVar('c_y')
        c_1 = TegVar('c_1')

        bindings = {c_xy: 1.2, c_x: 0.1, c_y: 0.1, c_1: 0, t: 0.5}
        # Derivative of threshold only.
        integral = Teg(self.near_zero, self.one,
                       Teg(self.near_zero, self.one,
                           IfElse(c_xy * x * y + c_x * x + c_y * y + c_1 < -t, 1, 0), y
                           ), x
                       )

        fd_d_t = finite_difference(integral, t, bindings=bindings)
        fd_d_cxy = finite_difference(integral, c_xy, bindings=bindings)
        fd_d_cx = finite_difference(integral, c_x, bindings=bindings)
        fd_d_cy = finite_difference(integral, c_y, bindings=bindings)
        fd_d_c1 = finite_difference(integral, c_1, bindings=bindings)

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t, c_xy, c_x, c_y, c_1])
        d_t_expr = reduce_to_base(d_t_expr)

        check_nested_lists(self, evaluate(d_t_expr, num_samples=1000, bindings=bindings),
                           [fd_d_t, fd_d_cxy, fd_d_cx, fd_d_cy, fd_d_c1], places=2)

    def test_bilinear_interpolation(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        c00 = Var('c00')
        c01 = Var('c01')
        c10 = Var('c10')
        c11 = Var('c11')

        bilinear_lerp = (c00 * (1 - x) + c01 * (x)) * (1 - y) +\
                        (c10 * (1 - x) + c11 * (x)) * (y)

        # Derivative of threshold only.
        integral = Teg(self.near_zero, self.near_one,
                       Teg(self.near_zero, self.near_one,
                           IfElse(bilinear_lerp > t, 1, 0), y
                           ), x
                       )
        bindings = {c00: 0.1, c11: 0.1, c01: 0.9, c10: 0.9, t: 0.55}

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t, c00, c01, c10, c11])
        """
        fd_d_t = finite_difference(integral, t, bindings=bindings)
        fd_d_c00 = finite_difference(integral, c00, bindings=bindings)
        fd_d_c01 = finite_difference(integral, c01, bindings=bindings)
        fd_d_c10 = finite_difference(integral, c10, bindings=bindings)
        fd_d_c11 = finite_difference(integral, c11, bindings=bindings)
        """

        (fd_d_t, fd_d_c00, fd_d_c01, fd_d_c10, fd_d_c11) = \
        [-2.549875, 0.5577500000000026, 0.7168749999999987, 0.7167500000000021, 0.5577500000000026]

        d_t_expr = simplify(reduce_to_base(simplify(d_t_expr)))

        check_nested_lists(self, evaluate(d_t_expr, num_samples=1000, bindings=bindings),
                           [fd_d_t, fd_d_c00, fd_d_c01, fd_d_c10, fd_d_c11], places=2)

    def test_bicubic_interpolation(self):
        x = TegVar('x')
        y = TegVar('y')
        t = Var('t')

        c00 = Var('c00')
        c01 = Var('c01')
        c10 = Var('c10')
        c11 = Var('c11')

        map_x_smoothstep, x_ = smoothstep(x)
        map_y_smoothstep, y_ = smoothstep(y)

        bicubic_lerp = (c00 * (1 - x_) + c01 * (x_)) * (1 - y_) +\
                       (c10 * (1 - x_) + c11 * (x_)) * (y_)

        # Derivative of threshold only.
        integral = Teg(self.near_zero, self.near_one,
                       Teg(self.near_zero, self.near_one,
                           map_x_smoothstep(map_y_smoothstep(
                                IfElse(bicubic_lerp > t, 1, 0)
                            )), y
                           ), x
                       )
        bindings = {c00: 0.1, c11: 0.1, c01: 0.9, c10: 0.9, t: 0.55}

        _, d_t_expr = reverse_deriv(integral, Tup(Const(1)), output_list=[t, c00, c01, c10, c11])

        integral = reduce_to_base(integral)
        fd_d_t = finite_difference(integral, t, bindings=bindings, num_samples=20000, delta=0.002)
        fd_d_c00 = finite_difference(integral, c00, bindings=bindings, num_samples=20000, delta=0.002)
        fd_d_c01 = finite_difference(integral, c01, bindings=bindings, num_samples=20000, delta=0.002)
        fd_d_c10 = finite_difference(integral, c10, bindings=bindings, num_samples=20000, delta=0.002)
        fd_d_c11 = finite_difference(integral, c11, bindings=bindings, num_samples=20000, delta=0.002)

        d_t_expr = simplify(reduce_to_base(simplify(d_t_expr)))

        check_nested_lists(self, evaluate(d_t_expr, num_samples=3000, bindings=bindings),
                           [fd_d_t, fd_d_c00, fd_d_c01, fd_d_c10, fd_d_c11], places=1)


class PolarMapTests(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_polar_map_simple(self):
        x = TegVar('x')
        y = TegVar('y')
        r = TegVar('r')

        # Constant square
        integral = Teg(self.zero, self.one * 2,
                       Teg(self.zero, self.one * 2,
                           polar_2d_map(Const(1), x=x, y=y, r=r), x
                           ), y
                       )

        self.assertAlmostEqual(evaluate(reduce_to_base(integral), num_samples=1000), 4.0, places=2)

        # Area of a unit circle.
        integral = Teg(-self.one, self.one,
                       Teg(-self.one, self.one,
                           polar_2d_map(IfElse(r < 1, self.one, self.zero), x=x, y=y, r=r), x
                           ), y
                       )

        self.assertAlmostEqual(evaluate(reduce_to_base(integral), num_samples=1000), np.pi, places=2)

    def test_polar_map_delta(self):
        x = TegVar('x')
        y = TegVar('y')
        r = TegVar('r')

        radius_value = 0.5
        radius = Var('radius', radius_value)
        # Delta along circumference of circle.
        x_low, x_high = Var('x_lb', -1), Var('x_ub', 1)
        y_low, y_high = Var('y_lb', -1), Var('y_ub', 1)

        area_integral = Teg(y_low, y_high,
                            Teg(x_low, x_high,
                                polar_2d_map(IfElse(r < radius, self.one, self.zero), x=x, y=y, r=r), x
                                ), y
                            )

        _, rev_dintegral = reverse_deriv(area_integral, output_list=[radius])
        fwd_dintegral = fwd_deriv(area_integral, [(radius, 1)])
        area_integral = simplify(reduce_to_base(area_integral))

        rev_dintegral = simplify(reduce_to_base(rev_dintegral))
        fwd_dintegral = simplify(reduce_to_base(fwd_dintegral))

        # Full circumference.
        self.assertAlmostEqual(evaluate(rev_dintegral,
                                        {x_low: -1, x_high: 1,
                                         y_low: -1, y_high: 1},
                                        num_samples=1000,
                                        ), 2 * np.pi * radius_value, places=2)
        self.assertAlmostEqual(evaluate(fwd_dintegral,
                                        {x_low: -1, x_high: 1,
                                         y_low: -1, y_high: 1},
                                        num_samples=1000,
                                        ), 2 * np.pi * radius_value, places=2)

        # Semi-circle
        self.assertAlmostEqual(evaluate(rev_dintegral,
                                        {x_low: 0, x_high: 1,
                                         y_low: -1, y_high: 1},
                                        num_samples=1000,
                                        ), np.pi * radius_value, places=2)
        self.assertAlmostEqual(evaluate(fwd_dintegral,
                                        {x_low: 0, x_high: 1,
                                         y_low: -1, y_high: 1},
                                        num_samples=1000,
                                        ), np.pi * radius_value, places=2)

        # Quarter circle
        self.assertAlmostEqual(evaluate(rev_dintegral,
                                        {x_low: 0, x_high: 1,
                                         y_low: -1, y_high: 0},
                                        num_samples=1000,
                                        ), 0.5 * np.pi * radius_value, places=2)
        self.assertAlmostEqual(evaluate(fwd_dintegral,
                                        {x_low: 0, x_high: 1,
                                         y_low: -1, y_high: 0},
                                        num_samples=1000,
                                        ), 0.5 * np.pi * radius_value, places=2)

        # Mini-pixel on corner
        y_ub = -0.2
        y_lb = -0.7
        x_ub = radius_value + 0.2
        x_lb = radius_value - 0.2

        d_area = finite_difference(area_integral, var=radius, delta=0.002, num_samples=10000, 
                                   bindings={x_low: x_lb, x_high: x_ub,
                                             y_low: y_lb, y_high: y_ub})
        # print(d_area)
        self.assertAlmostEqual(evaluate(rev_dintegral,
                                        {x_low: x_lb, x_high: x_ub,
                                         y_low: y_lb, y_high: y_ub},
                                        num_samples=1000,
                                        ), d_area, places=2)
        self.assertAlmostEqual(evaluate(fwd_dintegral,
                                        {x_low: x_lb, x_high: x_ub,
                                         y_low: y_lb, y_high: y_ub},
                                        num_samples=1000,
                                        ), d_area, places=2)


if __name__ == '__main__':
    unittest.main()
