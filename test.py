import unittest

from integrable_program import TegIntegral, TegVariable, TegConditional, TegConstant
from deriv import deriv
from back_deriv import back_deriv


class TestArithmetic(unittest.TestCase):

    def test_linear(self):
        x = TegVariable('x', 1)
        three_x = x + x + x
        self.assertAlmostEqual(three_x.eval(), 3, places=3)

    def test_multiply(self):
        x = TegVariable('x', 2)
        cube = x * x * x
        self.assertAlmostEqual(cube.eval(), 8, places=3)

    def test_polynomial(self):
        x = TegVariable('x', 2)
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(poly.eval(), 14, places=3)


class TestIntegrations(unittest.TestCase):

    def test_integrate_linear(self):
        a, b = 0, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x, x)
        self.assertAlmostEqual(integral.eval(), 0.5, places=3)

    def test_integrate_sum(self):
        a, b = 0, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} 2x dx
        integral = TegIntegral(a, b, x + x, x)
        self.assertAlmostEqual(integral.eval(), 1, places=3)

    def test_integrate_product(self):
        a, b = -1, 1
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x^2 dx
        integral = TegIntegral(a, b, x * x, x)
        self.assertAlmostEqual(integral.eval(), 2 / 3, places=1)

    def test_integrate_poly(self):
        a, b = -1, 2
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(integral.eval(1000), 11.25, places=3)

    def test_integrate_poly_poly_bounds(self):
        a, b = -1, 2
        x = TegVariable('x')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # int_{a*a}^{b} x dx
        integral = TegIntegral(a * a + a + a, b * b + a + a, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(integral.eval(1000), 11.25, places=3)


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
        self.assertAlmostEqual(integral.eval(), 0.5, places=3)

    def test_nested_integrals_different_variable(self):
        a, b = 0, 1
        x = TegVariable('x')
        t = TegVariable('t')
        a = TegVariable('a', a)
        b = TegVariable('b', b)
        # \int_0^1 x * \int_0^1 t dt dx
        body = TegIntegral(a, b, t, t)
        integral = TegIntegral(a, b, x * body, x)
        self.assertAlmostEqual(integral.eval(), 0.25, places=3)

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
        self.assertAlmostEqual(integral.eval(), 0, places=3)


class TestConditionals(unittest.TestCase):

    def test_basic_branching(self):
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        cond = TegConditional(x, 0, a, b)

        # if(x < c) 0 else 1 at x=-1
        cond.bind_variable('x', TegVariable('d', -1))
        self.assertEqual(cond.eval(), 0)

        # if(x < 0) 0 else 1 at x=1
        cond.bind_variable('x', b)
        self.assertEqual(cond.eval(ignore_cache=True), 1)

    def test_integrate_branch(self):
        a = TegVariable('a', -1)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        d = TegVariable('d', 0)
        # if(x < c) 0 else 1
        body = TegConditional(x, 0, d, b)
        integral = TegIntegral(a, b, body, x)

        # int_{a=-1}^{b=1} (if(x < c) 0 else 1) dx
        self.assertAlmostEqual(integral.eval(), 1, places=3)

    def test_branch_in_cond(self):
        # cond.bind_variable('x', b)
        # self.assertEqual(cond.eval(), 1)
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x', -1)
        t = TegVariable('t')
        # if(x < c) 0 else 1
        upper = TegConditional(x, 0, a, b)
        integral = TegIntegral(a, upper, t, t)

        # int_{a=0}^{if(x < c) 0 else 1} t dt
        self.assertEqual(integral.eval(), 0)


class TestForwardDerivatives(unittest.TestCase):

    def test_deriv_basics(self):
        x = TegVariable('x', 1)
        f = x + x
        deriv_expr = deriv(f, {'x': 1})
        # df(x=1)/dx
        self.assertEqual(deriv_expr.eval(), 2)

        f = x * x
        # df(x=1)/dx
        deriv_expr = deriv(f, {'x': 1})
        self.assertEqual(deriv_expr.eval(), 2)

    def test_deriv_poly(self):
        x = TegVariable('x', 1)
        c1 = TegConstant(1)
        c2 = TegConstant(3)
        f = c1 * x**3 + c2 * x**2 + c2 * x + c1
        # df(x=1)/dx
        deriv_expr = deriv(f, {'x': 1})
        self.assertEqual(deriv_expr.eval(), 12)

        deriv_expr = deriv(f, {'x': 1})
        deriv_expr.bind_variable('x', -1)
        # df(x=-1)/dx
        self.assertEqual(deriv_expr.eval(ignore_cache=True), 0)

    def test_deriv_branch(self):
        x = TegVariable('x')
        theta = TegVariable('theta', 1)
        f = TegConditional(x, 0, theta, theta * theta)

        deriv_expr = deriv(f, {'theta': 1})
        x.bind_variable('x', 1)
        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(deriv_expr.eval(), 2)

        x.bind_variable('x', -1)
        self.assertAlmostEqual(deriv_expr.eval(ignore_cache=True), 1)

    def test_deriv_integral(self):
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        theta = TegVariable('theta', 5)
        f = TegIntegral(a, b, x * theta, x)

        deriv_expr = deriv(f, {'theta': 1})

        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(deriv_expr.eval(), 0.5)

    def test_deriv_integral_branch_poly(self):
        a = TegConstant(name='a', value=0)
        b = TegConstant(name='b', value=1)

        theta1 = TegVariable('theta1', -1)
        theta2 = TegVariable('theta2', 1)
        # g = \int_a^b if(x<0.5) x*theta1 else 1 dx
        # h = \int_a^b x*theta2 + x^2theta1^2 dx
        x = TegVariable('x')
        body = TegConditional(x, 0.5, x * theta1, b)
        g = TegIntegral(a, b, body, x)
        h = TegIntegral(a, b, x * theta2 + x**2 * theta1**2, x)

        y = TegVariable('y', 0)
        # if(y < 1) g else h
        f = TegConditional(y, 1, g, h)

        # df/d(theta1)
        deriv_expr = deriv(f, {'theta1': 1, 'theta2': 0})
        self.assertAlmostEqual(deriv_expr.eval(), 0.125, places=3)

        # df/d(theta2)
        deriv_expr = deriv(f, {'theta1': 0, 'theta2': 1})
        self.assertAlmostEqual(deriv_expr.eval(ignore_cache=True), 0)

        f.bind_variable('y', 2)
        deriv_expr = deriv(f, {'theta1': 1, 'theta2': 0})
        self.assertAlmostEqual(deriv_expr.eval(ignore_cache=True), -2/3, places=3)

        deriv_expr = deriv(f, {'theta1': 0, 'theta2': 1})
        self.assertAlmostEqual(deriv_expr.eval(ignore_cache=True), 1/2, places=3)


class TestBackwardDerivatives(unittest.TestCase):

    def test_deriv_basics(self):
        x = TegVariable('x', 1)
        f = x + x
        deriv_dict = back_deriv(f, 1)
        # df(x=1)/dx
        self.assertEqual(deriv_dict['dx'].eval(), 2)

        f = x * x
        # df(x=1)/dx
        deriv_dict = back_deriv(f, 1)
        self.assertEqual(deriv_dict['dx'].eval(), 2)

    def test_deriv_poly(self):
        x = TegVariable('x', 1)
        c1 = TegConstant(1)
        c2 = TegConstant(3)
        f = c1 * x**3 + c2 * x**2 + c2 * x + c1
        # df(x=1)/dx
        deriv_dict = back_deriv(f, 1)
        self.assertEqual(deriv_dict['dx'].eval(), 12)

        deriv_dict = back_deriv(f, 1)
        deriv_dict['dx'].bind_variable('x', -1)
        # df(x=-1)/dx
        self.assertEqual(deriv_dict['dx'].eval(ignore_cache=True), 0)

    def test_deriv_branch(self):
        x = TegVariable('x')
        theta = TegVariable('theta', 1)
        f = TegConditional(x, 0, theta, theta * theta)

        deriv_expr = back_deriv(f)
        x.bind_variable('x', 1)
        # deriv(if (x<0) theta else theta^2)
        self.assertAlmostEqual(deriv_expr['dtheta'].eval(), 2)

        x.bind_variable('x', -1)
        self.assertAlmostEqual(deriv_expr['dtheta'].eval(ignore_cache=True), 1)

    def test_deriv_integral(self):
        a = TegVariable('a', 0)
        b = TegVariable('b', 1)

        x = TegVariable('x')
        theta = TegVariable('theta', 5)
        f = TegIntegral(a, b, x * theta, x)

        deriv_expr = back_deriv(f)

        # deriv(int_{a=0}^{1} x*theta dx)
        self.assertAlmostEqual(deriv_expr['dtheta'].eval(), 0.5)

    def test_deriv_integral_branch_poly(self):
        a = TegConstant(name='a', value=0)
        b = TegConstant(name='b', value=1)

        theta1 = TegVariable('theta1', -1)
        theta2 = TegVariable('theta2', 1)
        # g = \int_a^b if(x<0.5) x*theta1 else 1 dx
        # h = \int_a^b x*theta2 + x^2theta1^2 dx
        x = TegVariable('x')
        body = TegConditional(x, 0.5, x * theta1, b)
        g = TegIntegral(a, b, body, x)
        h = TegIntegral(a, b, x * theta2 + x**2 * theta1**2, x)

        y = TegVariable('y', 0)
        # if(y < 1) g else h
        f = TegConditional(y, 1, g, h)

        # df/d(theta1)
        deriv_expr = back_deriv(f)
        self.assertAlmostEqual(deriv_expr['dtheta1'].eval(), 0.125, places=3)
        self.assertAlmostEqual(deriv_expr['dtheta2'].eval(ignore_cache=True), 0)

        f.bind_variable('y', 2)
        deriv_expr = back_deriv(f)
        self.assertAlmostEqual(deriv_expr['dtheta1'].eval(ignore_cache=True), -2/3, places=3)
        self.assertAlmostEqual(deriv_expr['dtheta2'].eval(ignore_cache=True), 1/2, places=3)

# TODO: Implement derivatives
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) x
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) t


if __name__ == '__main__':
    unittest.main()