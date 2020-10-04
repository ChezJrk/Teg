import unittest
from unittest import TestCase
import numpy as np

from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    Tup,
    LetIn,
)
from derivs import FwdDeriv, RevDeriv
import operator_overloads  # noqa: F401
from simplify import simplify
from compile import emit
import subprocess


def check_nested_lists(self, results, expected, places=7):
    for res, exp in zip(results, expected):
        if isinstance(res, (list, np.ndarray)):
            check_nested_lists(self, res, exp, places)
        else:
            t = (int, float, np.int64, np.float)
            err = f'Result {res} of type {type(res)} and expected {exp} of type {type(exp)}'
            assert isinstance(res, t) and isinstance(exp, t), err
            self.assertAlmostEqual(res, exp, places)

def runProgram(program, silent = False):
    prog_name, out_size = program
    proc = subprocess.Popen([prog_name], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err is not None:
        print(f"Error: {err}")
        print(f"Output: {out}")

    if out_size > 1:
        return [ float(line) for line in out.decode().split('\n')[:out_size] ]
    else:
        return float(out)

def compileProgram(program):
    fn_name, arglist, code, out_size = (program.name, program.arglist, program.code, program.size)

    for arg in arglist:
        assert arg.default is not None, f'Var {arg.name} does not have a default value. Such programs are not currently supported'

    # Build dummy main function
    if out_size == 1:
        main_code = f' \n' +\
                f'int main(int argc, char** argv){{\n' +\
                f'  std::cout << {fn_name}();\n' +\
                f'  return 0;\n' +\
                f'}}'
    else:
        main_code = f' \n' +\
                f'int main(int argc, char** argv){{\n' +\
                f'  {fn_name}_result s = {fn_name}();\n' +\
                f'  for(int i = 0; i < {out_size}; i++){{' +\
                f'      std::cout << s.o[i] << std::endl;\n' +\
                f'  }}' +\
                f'  return 0;\n' + \
                f'}}'
    

    # Include basic IO
    header_code = "#include <iostream>\n"

    all_code = header_code + code + main_code
    cppfile = open("/tmp/_teg_cpp_out.cpp", "w")
    cppfile.write(all_code)
    cppfile.close()

    proc = subprocess.Popen("g++ /tmp/_teg_cpp_out.cpp -o /tmp/_teg_cpp_out -O3", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err is not None:
        print(f"Error: {err}")
        print(f"Output: {out}")

    return "/tmp/_teg_cpp_out", out_size

def evaluate(expr: ITeg, num_samples = 100, ignore_cache = False):
    return runProgram(compileProgram(emit(expr, target = 'C', num_samples = num_samples)))

# TODO: Move everything to test.py

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

    def test_let(self):
        x = Var('x', 2)
        y = Var('y', 4)
        add = x + y

        rev_res = RevDeriv(add, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_res), [1.0, 1.0])

    def test_polynomial(self):
        x = Var('x', 2)
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(evaluate(poly), 14, places=3)

class TestConditionals(TestCase):

    def test_basic_branching(self):
        a = Var('a', 0)
        b = Var('b', 1)

        x = Var('x')
        cond = IfElse(x < 0, a, b)

        # if(x < c) 0 else 1 at x=-1
        cond.bind_variable(x, -1)
        self.assertEqual(evaluate(cond), 0)

        # if(x < 0) 0 else 1 at x=1
        cond.bind_variable(x, 1)
        self.assertEqual(evaluate(cond), 1)

    def test_integrate_branch(self):
        a = Var('a', -1)
        b = Var('b', 1)

        x = Var('x')
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
        t = Var('t')
        # if(x < c) 0 else 1
        upper = IfElse(x < 0, a, b)
        integral = Teg(a, upper, t, t)

        # int_{a=0}^{if(x < c) 0 else 1} t dt
        self.assertEqual(evaluate(integral), 0)


class TestIntegrations(TestCase):

    def test_integrate_linear(self):
        a, b = 0, 1
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x dx
        integral = Teg(a, b, x, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_integrate_sum(self):
        a, b = 0, 1
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} 2x dx
        integral = Teg(a, b, x + x, x)
        self.assertAlmostEqual(evaluate(integral), 1, places=3)

    def test_integrate_product(self):
        a, b = -1, 1
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x^2 dx
        integral = Teg(a, b, x * x, x)
        self.assertAlmostEqual(evaluate(integral), 2 / 3, places=1)

    def test_integrate_division(self):
        x = Var('x')
        # int_{a}^{b} x^2 dx
        integral = Teg(1, 2, 1 / x**2, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_integrate_poly(self):
        a, b = -1, 2
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a}^{b} x dx
        integral = Teg(a, b, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, 1000), 11.25, places=3)

    def test_integrate_poly_poly_bounds(self):
        a, b = -1, 2
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # int_{a*a}^{b} x dx
        integral = Teg(a * a + a + a, b * b + a + a, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(evaluate(integral, 1000), 11.25, places=3)


class TestNestedIntegrals(TestCase):

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
        x = Var('x')
        a = Var('a', a)
        b = Var('b', b)
        # \int_0^1 \int_0^1 x dx dx
        body = Teg(a, b, x, x)
        integral = Teg(a, b, body, x)
        self.assertAlmostEqual(evaluate(integral), 0.5, places=3)

    def test_nested_integrals_different_variable(self):
        a, b = 0, 1
        x = Var('x')
        t = Var('t')
        a = Var('a', a)
        b = Var('b', b)
        # \int_0^1 x * \int_0^1 t dt dx
        body = Teg(a, b, t, t)
        integral = Teg(a, b, x * body, x)
        self.assertAlmostEqual(evaluate(integral), 0.25, places=3)

    def test_integral_with_integral_in_bounds(self):
        a, b = 0, 1
        x = Var('x')
        t = Var('t')
        a = Var('a', a)
        b = Var('b', b)
        # \int_{\int_0^1 2x dx}^{\int_0^1 xdx + \int_0^1 tdt} x dx
        integral1 = Teg(a, b, x + x, x)
        integral2 = Teg(a, b, t, t) + Teg(a, b, x, x)

        integral = Teg(integral1, integral2, x, x)
        self.assertAlmostEqual(evaluate(integral), 0, places=3)

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
        x = Var('x')
        v = Tup(*[Const(i) for i in range(3)])
        res = Teg(Const(0), Const(1), v * x, x)
        res, expected = evaluate(res), [0, .5, 1]
        check_nested_lists(self, res, expected)

class VariableBranchConditionsTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_deriv_heaviside(self):
        x = Var('x')
        t = Var('t', 0.5)
        heaviside = IfElse(x < t, self.zero, self.one)
        integral = Teg(self.zero, self.one, heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1, places = 5)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), -1, places = 5)

        neg_heaviside = IfElse(x < t, self.one, self.zero)
        integral = Teg(self.zero, self.one, neg_heaviside, x)

        # d/dt int_{x=0}^{1} (x<t) ? 1 : 0
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 1, places = 5)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral), 1, places = 5)

    def test_deriv_heaviside_discontinuity_out_of_domain(self):
        x = Var('x')
        t = Var('t', 2)
        heaviside = IfElse(x < t, self.zero, self.one)
        integral = Teg(self.zero, self.one, heaviside, x)

        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

        t.bind_variable(t, -1)
        # d/dt int_{x=0}^1 (x<t) ? 0 : 1
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 0)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

    def test_deriv_scaled_heaviside(self):
        x = Var('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        body = (x + t) * heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 (x + t) * ((x<t) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -0.5)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), -0.5)

    def test_deriv_add_heaviside(self):
        x = Var('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        body = x + t + heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 x + t + ((x<t) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), 0, places = 5)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0, places = 5)
    
    def test_deriv_product_heavisides(self):
        x = Var('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse(x < t, Const(0), Const(1))
        flipped_heavyside_t = IfElse(x < t, Const(1), Const(0))
        body = heavyside_t * heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 0 : 1) * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), -1)

        body = flipped_heavyside_t * heavyside_t
        integral = Teg(Const(0), Const(1), body, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) ? 1 : 0) * ((x<t=0.5) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), 0)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), 0)

        t1 = Var('t1', 0.3)
        heavyside_t1 = IfElse(x < t1, Const(0), Const(1))
        body = heavyside_t * heavyside_t1
        integral = Teg(Const(0), Const(1), body, x)

        # D int_{x=0}^1 ((x<t=0.5) ? 0 : 1) * ((x<t1=0.3) ? 0 : 1)
        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral, ignore_cache=True), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), [-1, 0])
    
    def test_tzu_mao(self):
        # \int_{x=0}^1 x < t ?
        #   \int_{y=0}^1 x < t1 ?
        #     x * y : x * y^2 :
        #   \int_{y=0}^1 y < t2 ?
        #     x^2 * y : x^2 * y^2
        zero, one = Const(0), Const(1)
        x, y = Var('x'), Var('y')
        t, t1, t2 = Var('t', 0.5), Var('t1', 0.25), Var('t2', 0.75)
        if_body = Teg(zero, one, IfElse(x < t1, x * y, x * y * y), y)
        else_body = Teg(zero, one, IfElse(y < t2, x * x * y, x * x * y * y), y)
        integral = Teg(zero, one, IfElse(x < t, if_body, else_body), x)

        expected = [0.048, 0.041, 0.054]
        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), expected, places=2)

        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 0), (t2, 0)])
        dt = evaluate(fwd_dintegral, ignore_cache=True)

        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1), (t2, 0)])
        dt1 = evaluate(fwd_dintegral, ignore_cache=True)

        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 0), (t2, 1)])
        dt2 = evaluate(fwd_dintegral, ignore_cache=True)

        check_nested_lists(self, [dt, dt1, dt2], expected, places=2)

    def test_single_integral_example(self):
        # deriv(\int_{x=0}^1 (x < theta ? 1 : x * theta))
        zero, one = Const(0), Const(1)
        x, theta = Var('x'), Var('theta', 0.5)

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
        x, y = Var('x'), Var('y')

        body = IfElse(x < y, zero, one)
        integral = Teg(zero, one, body, x)
        body = RevDeriv(integral, Tup(Const(1)))
        double_integral = Teg(zero, one, body, y)
        self.assertAlmostEqual(evaluate(double_integral, num_samples=100), -1, places=1)

    def test_nested_integral_moving_discontinuity(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, y, t = Var('x'), Var('y'), Var('t', 0.5)

        body = Teg(zero, one, IfElse(x < t, zero, one), x)
        integral = Teg(zero, one, y * body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -0.5, places=3)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -0.5, places=3)

    def test_nested_discontinuity_integral(self):
        # deriv(\int_{y=0}^{1} y * \int_{x=0}^{1} (x<t) ? 0 : 1)
        zero, one = Const(0), Const(1)
        x, t1, t = Var('x'), Var('t1', 0.5), Var('t', 0.4)

        body = IfElse(t < t1, IfElse(x < t, zero, one), one)
        integral = Teg(zero, one, body, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=3)

        # TODO: Maybe make dt1 a term rather than just t
        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -1, places=3)

class MovingBoundaryTest(TestCase):

    def test_moving_upper_boundary(self):
        # deriv(\int_{x=0}^{t=1} xt)
        # 0.5 + 1 - 0 = 1.5
        zero = Const(0)
        x, t = Var('x'), Var('t', 1)
        integral = Teg(zero, t, x * t, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), 1.5)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 1.5)

    def test_moving_lower_boundary(self):
        # deriv(\int_{x=t=-1}^{1} xt)
        # 0 + 0 + 1 = 1
        one = Const(1)
        x, t = Var('x'), Var('t', -1)
        integral = Teg(t, one, x * t, x)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral), -1, places=1)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -1, places=1)

    def test_both_boundaries_moving(self):
        # deriv(\int_{x=y=0}^{z=1} 1)
        # = \int 0 + dz - dy
        one = Const(1)
        x, y, z = Var('x'), Var('y', 0), Var('z', 1)
        integral = Teg(y, z, one, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [1, -1])

        deriv_integral = FwdDeriv(integral, [(y, 1), (z, 0)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -1)

        deriv_integral = FwdDeriv(integral, [(y, 0), (z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 1)

    def test_nested_integral_boundaries_moving(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = Const(0), Const(1)
        x, y, z = Var('x'), Var('y'), Var('z', 1)

        body = Teg(y, z, y * z, x)
        integral = Teg(zero, one, body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 2/3, places=3)

        deriv_integral = FwdDeriv(integral, [(z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 2/3, places=3)

    def test_nested_integral_boundaries_moving_scaled(self):
        # deriv(\int_{y=0}^{1} \int_{x=y=0}^{z=1} y)
        # = \int 0 + dz - dy
        zero, one = Const(0), Const(1)
        x, y, z = Var('x'), Var('y'), Var('z', 1)

        body = Teg(y, z, y * z, x)
        integral = Teg(zero, one, y * body, y)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 5/12, places=3)

        deriv_integral = FwdDeriv(integral, [(z, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 5/12, places=3)

    def test_affine_moving_boundary(self):
        # deriv(\int_{x=a - b}^{a + b} 1)
        x, a, b = Var('x'), Var('a', 0), Var('b', 1)

        integral = Teg(a - b, a + b, 1, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [0, 2], places=3)

        deriv_integral = FwdDeriv(integral, [(a, 1), (b, 0)])
        # print(deriv_integral.deriv_expr)
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 0, places=3)

        deriv_integral = FwdDeriv(integral, [(a, 0), (b, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 2, places=3)

    def test_affine_moving_boundary_variable_body(self):
        # deriv(\int_{x=a - b}^{a + b} x)
        x, a, b = Var('x'), Var('a', 0), Var('b', 1)
        integral = Teg(a - b, a + b, x, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [2, 0], places=3)

        deriv_integral = FwdDeriv(integral, [(a, 1), (b, 0)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 2, places=3)

        deriv_integral = FwdDeriv(integral, [(a, 0), (b, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 0, places=3)


class PiecewiseAffineTest(TestCase):

    def test_and_same_location(self):
        x = Var('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse((x < t) & (x < t), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), -1)

    def test_or_same_location(self):
        x = Var('x')
        t = Var('t', 0.5)
        heavyside_t = IfElse((x < t) | (x < t), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(rev_dintegral, ignore_cache=True), -1)

    def test_and_different_locations(self):
        x = Var('x')
        t = Var('t', 0.5)
        t1 = Var('t1', 0.3)
        heavyside_t = IfElse((x < t) & (x < t1), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), [0, -1])

    def test_or_different_locations(self):
        x = Var('x')
        t = Var('t', 0.5)
        t1 = Var('t1', 0.3)
        heavyside_t = IfElse((x < t) | (x < t1), Const(0), Const(1))

        integral = Teg(0, 1, heavyside_t, x)
        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 1), (t1, 0)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)

        rev_dintegral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(rev_dintegral, ignore_cache=True), [-1, 0])

    def test_compound_exprs(self):
        x = Var('x')
        t = Var('t', 0.8)
        t1 = Var('t1', 0.3)
        t2 = Var('t2', 0.1)
        if_else = IfElse(((x < t) & (x < t1 + t1)) & (x < t2 + 1), Const(0), Const(1))

        integral = Teg(0, 1, if_else, x)

        # d/dt int_{x=0}^1 ((x<t=0.5) & (x<t=0.5) ? 1 : 0)
        fwd_dintegral = FwdDeriv(integral, [(t, 0), (t1, 1), (t2, 0)])
        self.assertAlmostEqual(evaluate(fwd_dintegral), -1)


class AffineConditionsTest(TestCase):

    def setUp(self):
        self.zero, self.one = Const(0), Const(1)

    def test_affine_condition_simple(self):
        x, t = Var('x'), Var('t', 0.25)
        cond = IfElse(x < Const(2) * t, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), -2)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), -2)

    def test_affine_condition_with_constants(self):
        x, t = Var('x'), Var('t', 7/4)
        cond = IfElse(x + Const(2) * t - Const(4) < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        self.assertAlmostEqual(evaluate(deriv_integral), 2)

        deriv_integral = FwdDeriv(integral, [(t, 1)])
        self.assertAlmostEqual(evaluate(deriv_integral, ignore_cache=True), 2)

    def test_affine_condition_multivariable(self):
        # \int_{x = [0, 1]} (x - 2 * t1 + 3 * t2 ? 0 : 1)
        x, t1, t2 = Var('x'), Var('t1', 3/4), Var('t2', 1/3)
        cond = IfElse(x - Const(2) * t1 + Const(3) * t2 < 0, self.zero, self.one)
        integral = Teg(self.zero, self.one, cond, x)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(deriv_integral), [-2, 3])

        deriv_integral = FwdDeriv(integral, [(t1, 1), (t2, 0)])
        self.assertEqual(evaluate(deriv_integral, ignore_cache=True), -2)

        deriv_integral = FwdDeriv(integral, [(t1, 0), (t2, 1)])
        self.assertEqual(evaluate(deriv_integral, ignore_cache=True), 3)

    def test_affine_condition_multivariable_multiintegral(self):
        x, y = Var('x'), Var('y')
        t1, t2 = Var('t1', 1/4), Var('t2', 1)
        cond = IfElse(x - y + Const(2) * t1 + Const(3) * t2 + Const(-3) < 0, self.one, self.zero)
        body = Teg(self.zero, self.one, cond, y)
        integral = Teg(self.zero, self.one, body, x)
        # int_{x=[0, 1]}
        #     int_{y=[0, 1]}
        #         ((x - y + 2 * t1=0.25 + 3 * t2=1 - 3 < 0) ? 1 : 0)

        deriv_integral = RevDeriv(integral, Tup(Const(1)))
        check_nested_lists(self, evaluate(simplify(deriv_integral)), [1, 1.5])

        deriv_integral = FwdDeriv(integral, [(t1, 1), (t2, 0)])
        self.assertAlmostEqual(evaluate(simplify(deriv_integral)), 1)

        deriv_integral = FwdDeriv(integral, [(t1, 0), (t2, 1)])
        self.assertAlmostEqual(evaluate(simplify(deriv_integral)), 1.5)

if __name__ == '__main__':
    unittest.main()
