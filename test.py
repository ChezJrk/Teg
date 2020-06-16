import unittest
import torch

from integrable_program import TegIntegral, TegVariable, TegConditional


class TestArithmetic(unittest.TestCase):

    def test_linear(self):
        x = TegVariable("x", 1)
        three_x = x + x + x
        self.assertAlmostEqual(three_x.eval(), 3, places=3)

    def test_multiply(self):
        x = TegVariable("x", 2)
        cube = x * x * x
        self.assertAlmostEqual(cube.eval(), 8, places=3)

    def test_polynomial(self):
        x = TegVariable("x", 2)
        poly = x * x * x + x * x + x
        self.assertAlmostEqual(poly.eval(), 14, places=3)


class TestIntegrations(unittest.TestCase):

    def test_integrate_linear(self):
        a, b = 0, 1
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x, x)
        self.assertAlmostEqual(integral.eval(), 0.5, places=3)

    def test_integrate_sum(self):
        a, b = 0, 1
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # int_{a}^{b} 2x dx
        integral = TegIntegral(a, b, x + x, x)
        self.assertAlmostEqual(integral.eval(), 1, places=3)

    def test_integrate_product(self):
        a, b = -1, 1
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # int_{a}^{b} x^2 dx
        integral = TegIntegral(a, b, x * x, x)
        self.assertAlmostEqual(integral.eval(), 2 / 3, places=1)

    def test_integrate_poly(self):
        a, b = -1, 2
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # int_{a}^{b} x dx
        integral = TegIntegral(a, b, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(integral.eval(1000), 11.25, places=3)

    def test_integrate_poly_poly_bounds(self):
        a, b = -1, 2
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # int_{a*a}^{b} x dx
        integral = TegIntegral(a * a + a + a, b * b + a + a, x * x * x + x * x + x * x + x, x)
        self.assertAlmostEqual(integral.eval(1000), 11.25, places=3)


class TestNestedIntegrals(unittest.TestCase):

    def test_nested_integrals_with_variable_bounds(self):
        a, b = -1, 1
        x = TegVariable("x")
        t = TegVariable("t")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # \int_-1^1 \int_{t}^{t+1} xt dx dt
        body = TegIntegral(t, t + b, x * t, x)
        integral = TegIntegral(a, b, body, t)
        self.assertAlmostEqual(integral.eval(100), 2 / 3, places=3)

    def test_nested_integrals_same_variable(self):
        a, b = 0, 1
        x = TegVariable("x")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # \int_0^1 \int_0^1 x dx dx
        body = TegIntegral(a, b, x, x)
        integral = TegIntegral(a, b, body, x)
        self.assertAlmostEqual(integral.eval(), 0.5, places=3)

    def test_nested_integrals_different_variable(self):
        a, b = 0, 1
        x = TegVariable("x")
        t = TegVariable("t")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # \int_0^1 x * \int_0^1 t dt dx
        body = TegIntegral(a, b, t, t)
        integral = TegIntegral(a, b, x * body, x)
        self.assertAlmostEqual(integral.eval(), 0.25, places=3)
    
    def test_integral_with_integral_in_bounds(self):
        a, b = 0, 1
        x = TegVariable("x")
        t = TegVariable("t")
        a = TegVariable("a", a)
        b = TegVariable("b", b)
        # \int_{\int_0^1 2x dx}^{\int_0^1 xdx + \int_0^1 tdt} x dx
        integral1 = TegIntegral(a, b, x + x, x)
        integral2 = TegIntegral(a, b, t, t) + TegIntegral(a, b, x, x)

        integral = TegIntegral(integral1, integral2, x, x)
        self.assertAlmostEqual(integral.eval(), 0, places=3)


class TestConditionals(unittest.TestCase):

    def test_basic_branching(self):
        a = TegVariable("a", 0)
        b = TegVariable("b", 1)

        x = TegVariable("x")
        cond = TegConditional(x, 0, a, b)

        # if(x < c) 0 else 1 at x=-1
        cond.bind_variable("x", TegVariable("d", -1))
        self.assertEqual(cond.eval(), 0)

        # if(x < 0) 0 else 1 at x=1
        cond.bind_variable("x", b)
        self.assertEqual(cond.eval(ignore_cache=True), 1)

    def test_integrate_branch(self):
        a = TegVariable("a", -1)
        b = TegVariable("b", 1)

        x = TegVariable("x")
        d = TegVariable("d", 0)
        # if(x < c) 0 else 1
        body = TegConditional(x, 0, d, b)
        integral = TegIntegral(a, b, body, x)

        # int_{a=-1}^{b=1} (if(x < c) 0 else 1) dx
        self.assertAlmostEqual(integral.eval(), 1, places=3)

    def test_branch_in_cond(self):
        # cond.bind_variable("x", b)
        # self.assertEqual(cond.eval(), 1)
        a = TegVariable("a", 0)
        b = TegVariable("b", 1)

        x = TegVariable("x", -1)
        t = TegVariable("t")
        # if(x < c) 0 else 1
        upper = TegConditional(x, 0, a, b)
        integral = TegIntegral(a, upper, t, t)

        # int_{a=0}^{if(x < c) 0 else 1} t dt
        self.assertEqual(integral.eval(), 0)


class TestDerivatives(unittest.TestCase):

    def test_deriv_basics(self):
        x = TegVariable("x", torch.tensor(1., requires_grad=True))
        y = x + x
        y.eval()
        y.backward()
        self.assertEqual(float(x.grad), 2)

        # Wipe the old gradient.
        x.value.grad = None
        y = x * x
        y.eval(ignore_cache=True)
        y.backward()
        self.assertEqual(float(x.grad), 2)

    # def test_deriv_integral(self):
    #     a = TegVariable("a", -1)
    #     b = TegVariable("b", 1)

    #     x = TegVariable("x")
    #     theta = TegVariable("theta", torch.tensor(1., requires_grad=True))
    #     y = TegIntegral(a, b, x * theta, x)

    #     # deriv(int_{a=-1}^{1} x*theta dx)
    #     y.eval(ignore_cache=True)
    #     y.backward()


# TODO: Implement derivatives
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) x
# deriv (\int_-1^1 \int_{t}^{t+1} xt dx dt) t

# Maybe use nn.Module rather than jerry-rigging gradients


if __name__ == '__main__':
    unittest.main()