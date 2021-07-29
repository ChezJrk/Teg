from teg import ITeg

from .c_eval import C_EvalMode, C_EvalMode_PyBind
from .numpy_eval import Numpy_EvalMode

BACKENDS = {
    'C': (C_EvalMode, {}),
    'C_PyBind': (C_EvalMode_PyBind, {}),
    'numpy': (Numpy_EvalMode, {})
}


def evaluate(expr: ITeg, bindings=None, backend=None, **kwargs) -> float:
    """Evaluates expressions with any of various backgrounds.

    Args:
        expr: An expression to be evaluated.
        bindings: A mapping from variable names to the values.
        backend: The supported backends are 'C', 'C_PyBind, 'numpy'.
        kwargs: Extra flags to be passed to the backend.

    Returns:
        Teg: The forward fwd_derivative expression.
    """

    if not hasattr(expr, 'mode_cache'):
        setattr(expr, 'mode_cache', {})
        setattr(expr, 'mode_options', kwargs)

    bindings = {} if bindings is None else bindings
    # NOTE: Replace with priorities if we use more modes later..
    backend = 'C' if backend is None else backend

    # TODO: This doesn't look like an airtight cache check..
    if backend not in expr.mode_cache.keys() or\
       not all([(key, value) in expr.mode_options.items() for key, value in kwargs.items()]): 
        backend_cls, backend_kwargs = BACKENDS[backend]
        mode = backend_cls(expr, **kwargs, **backend_kwargs)
        expr.mode_cache[backend] = mode
        expr.mode_options = kwargs

    return expr.mode_cache[backend].eval(bindings=bindings)
