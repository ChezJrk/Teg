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
    
    print(code)
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

    proc = subprocess.Popen("g++ /tmp/_teg_cpp_out.cpp -o /tmp/_teg_cpp_out", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err is not None:
        print(f"Error: {err}")
        print(f"Output: {out}")

    return "/tmp/_teg_cpp_out", out_size

def evaluate(expr: ITeg, num_samples = 50):
    return runProgram(compileProgram(emit(expr, target = 'C', num_samples = num_samples)))

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

if __name__ == '__main__':
    unittest.main()
