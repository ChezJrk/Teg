from teg import ITeg

from .c_eval import C_EvalMode, C_EvalMode_PyBind
from .numpy_eval import Numpy_EvalMode

BACKENDS = {
    'C': (C_EvalMode, {}),
    'C_PyBind': (C_EvalMode_PyBind, {}),
    'numpy': (Numpy_EvalMode, {})
}
BACKEND_PRIORITY_ORDER = ['C', 'numpy', 'C_PyBind']

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
    
    # Pick the first available backend in the order 
    if backend is None:
        for _name in BACKEND_PRIORITY_ORDER:
            backend_cls, backend_kwargs = BACKENDS[_name]
            try:
                backend_cls.assert_is_available(**backend_kwargs)
                backend = _name
                break
            except AssertionError as e:
                print(f'WARNING: {e}.'
                        f' To suppress these warnings, explicitly provide the backend parameter,' 
                        f' for example: evaluate(expr, backend="C")')
                continue

    # TODO: This doesn't look like an airtight cache check..
    if backend not in expr.mode_cache.keys() or\
       not all([(key, value) in expr.mode_options.items() for key, value in kwargs.items()]): 
        backend_cls, backend_kwargs = BACKENDS[backend]
        backend_cls.assert_is_available(**backend_kwargs)
        mode = backend_cls(expr, **kwargs, **backend_kwargs)
        expr.mode_cache[backend] = mode
        expr.mode_options = kwargs

    return expr.mode_cache[backend].eval(bindings=bindings)
