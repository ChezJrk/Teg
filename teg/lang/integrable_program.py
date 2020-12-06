from typing import Optional, List, Dict, Any
import operator
import numpy as np


def try_making_teg_const(x):
    if type(x) in (int, float, np.int64, np.float, np.float64):
        x = Const(x)
    return x


class ITeg:

    def __init__(self, children: List):
        super(ITeg, self).__init__()

        self.children = children

        for child in self.children:
            assert isinstance(child, ITeg), f'Non-ITeg expression {child} cannot be used in graph.'

        self.value = None

    def bind_variable(self, var: 'Var', value: Optional[float] = None) -> None:
        for child in self.children:
            child.bind_variable(var, value)


class PiecewiseAffine(ITeg):
    pass


class Var(PiecewiseAffine):
    global_uid = 0

    def __init__(self, name: str, value: Optional[float] = None, uid: Optional[int] = None):
        super(Var, self).__init__(children=[])
        self.name = name
        self.value = value
        if uid is None:
            self.uid = Var.global_uid
            Var.global_uid += 1
        else:
            self.uid = uid

    def bind_variable(self, var: 'Var', value: Optional[float] = None) -> None:
        if (self.name, self.uid) == (var.name, var.uid):
            self.value = value


class Const(Var):

    def __init__(self, value: Optional[float], name: str = ''):
        super(Const, self).__init__(name=name, value=value)

    def bind_variable(self, var: Var, value: Optional[float] = None) -> None:
        pass


class SmoothFunc(ITeg):
    """
        Arbitrary smooth function of one variable with symbolically defined derivatives.
        SmoothFunc is an abstract class that children must implement.
    """
    def __init__(self, expr: ITeg, name: str = 'SmoothFunc'):
        super(SmoothFunc, self).__init__(children=[expr])
        self.expr = expr
        self.name = name

    def fwd_deriv(self, in_deriv_expr: ITeg) -> ITeg:
        raise NotImplementedError

    def rev_deriv(self, out_deriv_expr: ITeg) -> ITeg:
        raise NotImplementedError

    def operation(self, in_value):
        raise NotImplementedError


class TegVar(Var):
    def __init__(self, name: str = '', uid: Optional[int] = None):
        super(TegVar, self).__init__(name=name, uid=uid)

    def bind_variable(self, var: Var, value: Optional[float] = None) -> None:
        if (self.name, self.uid) == (var.name, var.uid):
            self.value = value


class Add(PiecewiseAffine):
    name = 'add'
    operation = operator.add


class Mul(PiecewiseAffine):
    name = 'mul'
    operation = operator.mul


class Invert(ITeg):

    def __init__(self, child):
        super(Invert, self).__init__(children=[child])
        self.child = child


class Teg(ITeg):

    def __init__(self, lower: Var, upper: Var, body: ITeg, dvar: TegVar):
        super(Teg, self).__init__(children=[try_making_teg_const(e) for e in (lower, upper, body)])
        self.lower, self.upper, self.body = self.children
        assert isinstance(dvar, TegVar), f'Can only integrate over TegVar variables. {dvar} is not a TegVar'
        self.dvar = dvar


class IfElse(ITeg):

    def __init__(self, cond: 'ITegBool', if_body: ITeg, else_body: ITeg):
        super(IfElse, self).__init__(children=[try_making_teg_const(e) for e in (if_body, else_body)])
        self.cond = cond
        self.if_body, self.else_body = self.children

    def bind_variable(self, var: 'Var', value: Optional[float] = None) -> None:
        self.cond.bind_variable(var, value)
        for child in self.children:
            child.bind_variable(var, value)


class Tup(ITeg):

    def __init__(self, *args):
        super(Tup, self).__init__(children=[try_making_teg_const(e) for e in args])


class LetIn(ITeg):

    def __init__(self,
                 new_vars: Tup,
                 new_exprs: Tup,
                 expr: ITeg):
        super(LetIn, self).__init__(children=[try_making_teg_const(e) for e in (expr, *new_exprs)])
        self.new_vars = new_vars
        self.new_exprs = self.children[1:]
        self.expr = self.children[0]


class Func(ITeg):

    def __init__(self, name, body, *args):
        super(LetIn, self).__init__(children=[try_making_teg_const(body)])
        self.name = name
        self.body = body
        self.args = args


class Ctx(dict):
    """Mapping from strings to TegVariables. """
    global_uid = 0

    def __init__(self, env: Optional[Dict[str, Any]] = None, parent: Optional['TegContext'] = None):
        self.uid = Ctx.global_uid
        Ctx.global_uid += 1
        super().update({} if env is None else env)
        self.parent = {} if parent is None else parent

    def __getitem__(self, key: str) -> Any:
        if key in self.env:
            return self.env[key]
        elif not self.parent:
            raise ValueError(f'No value for the variable "{key}" has been set.')
        else:
            return self.parent[key]


class ITegBool(ITeg):

    def __init__(self):
        self.value = None

    def bind_variable(self, var: 'Var', value: Optional[float] = None) -> None:
        self.left_expr.bind_variable(var, value)
        self.right_expr.bind_variable(var, value)


class Bool(ITegBool):

    def __init__(self, left_expr: PiecewiseAffine, right_expr: PiecewiseAffine, allow_eq: bool = False):
        super(Bool, self).__init__()
        self.left_expr = try_making_teg_const(left_expr)
        self.right_expr = try_making_teg_const(right_expr)
        self.allow_eq = allow_eq


class BinBool(ITegBool):

    def __init__(self, left_expr: ITegBool, right_expr: ITegBool):
        super(BinBool, self).__init__()
        self.left_expr = left_expr
        self.right_expr = right_expr


class And(BinBool):
    pass


class Or(BinBool):
    pass


true = Bool(Const(0), Const(1))
false = Bool(Const(1), Const(0))
