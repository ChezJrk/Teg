# Teg

Teg is a differentiable programming language that includes an integral primitive, which allows for soundly optimizing integrals with discontinuous integrands. This is a research artifact for the paper: **Systematically Optimizing Parametric Discontinuities**. This repository contains the core library implementation, while the applications can be found at [https://github.com/ChezJrk/teg_applications](https://github.com/ChezJrk/teg_applications). The applications include image stylization, fitting shader parameters, trajectory optimization, and optimizing physical designs.



## Installation Instructions
Teg requires Python 3.6+. To install Teg run:
```
git clone https://github.com/ChezJrk/Teg.git
cd Teg
pip install -e .
```

## Illustrative Example
A minimal illustrative example is:

![\frac{d}{dt} \int_{x = 0}^1 [x > t]](https://latex.codecogs.com/svg.latex?\frac{d}{dt}%20\int_{x%20=%200}^1%20[x%20%3E%20t])

This is the integral of a step discontinuity that jumps from 0 to 1 at ![](https://latex.codecogs.com/svg.latex?t).
If we set ![](https://latex.codecogs.com/svg.latex?t%20=%200.5), then the result is 1, but discretizing before computing the derivative as is standard in differentiable programming languages (e.g., PyTorch and TensorFlow) results in a derivative of 0. Our language correctly models the interaction between the integral and the parametric discontinuity. In our language the implementation for this simple function is:
```python
from teg import TegVar, Var, Teg, IfElse
from teg.derivs import FwdDeriv
from teg.eval.numpy_eval import evaluate

x, t = TegVar('x'), Var('t', 0.5)
expr = Teg(0, 1, IfElse(x > t, 0, 1), x)
deriv_expr = FwdDeriv(expr, [(t, 1)])
print(evaluate(deriv_expr)) # prints 1
```

## Code Structure
Our implementation is in the `teg` folder:
 - `derivs` has the implementation of the source-to-source derivative including code for computing the forward and reverse derivatives in `fwd_deriv.py` and `reverse_deriv.py` respectively. Supported discontinuous functions are in the `edge` folder.
 - `eval` and `include` have all of the code for evaluating expressions either by interpreting locally in Python or compiling to C.
 - `ir` includes an intermediate representation useful in compiling code down to C.
 - `lang` includes the main language primitives with the base language in `base.py` and `teg.py` and the extended language (that has the Dirac delta function) is in `extended.py`.
 - `maps` and `math` specifies basic math libraries.
 - `passes` includes source-to-source compilation passes. Notably, `reduce.py` has the lowering code from the external language to the internal language.

We have a `test` folder with all of the systems tests.

## Compiling to C
It is possible to compile Teg programs to C using `python3 -m teg --compile --target C [[FILE NAME]].py`. See `teg/__main__.py` for more options.
To specify C compilation flags, use `python3 -m teg --include-options`.
