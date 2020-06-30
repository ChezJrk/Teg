from typing import Optional
import operator


class Teg:

    def __init__(self, children, sign=1):
        super(Teg, self).__init__()
        self.children = children
        self.sign = sign
        self.value = None

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        [child.bind_variable(var_name, value) for child in self.children]

    def unbind_variable(self, var_name: str) -> None:
        self.bind_variable(var_name, None)


class TegVariable(Teg):

    def __init__(self, name: str, value: Optional[float] = None):
        super(TegVariable, self).__init__(children=[])
        self.name = name
        self.value = value

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        if self.name == var_name:
            self.value = value


class TegConstant(TegVariable):

    def __init__(self, value: Optional[float], name: str = ''):
        super(TegConstant, self).__init__(name='', value=value)
        self.value = value
        self.name = name

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        pass


class TegAdd(Teg):
    name = 'add'
    operation = operator.add


class TegMul(Teg):
    name = 'mul'
    operation = operator.mul


class TegIntegral(Teg):

    def __init__(self, lower: TegConstant, upper: TegConstant, body: Teg, dvar: TegVariable):
        super(TegIntegral, self).__init__(children=[lower, upper, body, dvar])
        self.lower = lower
        self.upper = upper
        self.body = body
        self.dvar = dvar

    def bind_variable(self, var_name: str, value: Optional[float]):
        # assert self.dvar.name == var_name, (f'The name variable for the infinitesimal "{self.dvar.name}" '
        # f'should be different than the variable "{var_name}" that is bound.')
        self.lower.bind_variable(var_name, value)
        self.upper.bind_variable(var_name, value)
        self.body.bind_variable(var_name, value)


class TegConditional(Teg):

    def __init__(self, var: TegVariable, const: TegConstant, if_body: Teg, else_body: Teg):
        super(TegConditional, self).__init__(children=[var, const, if_body, else_body])
        self.var = var
        self.const = const
        self.if_body = if_body
        self.else_body = else_body

    def bind_variable(self, var_name: str, value: Optional[float]):
        self.var.bind_variable(var_name, value)
        self.if_body.bind_variable(var_name, value)
        self.else_body.bind_variable(var_name, value)


class TegTuple(Teg):

    def __init__(self, *args):
        super(TegTuple, self).__init__(children=args)


class TegLetIn(Teg):

    def __init__(self,
                 new_vars: TegTuple,
                 new_exprs: TegTuple,
                 var: TegVariable,
                 expr: Teg):
        super(TegLetIn, self).__init__(children=[expr, *new_exprs])
        self.new_vars = new_vars
        self.new_exprs = new_exprs
        self.var = var
        self.expr = expr


class TegContext(dict):
    """Mapping from strings to TegVariables. """
    pass
