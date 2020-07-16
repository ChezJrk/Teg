from typing import Optional, List, Dict, Any
import operator


class Teg:

    def __init__(self, children: List):
        super(Teg, self).__init__()
        self.children = children
        self.value = None

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        [child.bind_variable(var_name, value) for child in self.children]

    def unbind_variable(self, var_name: str) -> None:
        self.bind_variable(var_name, None)


class TegVariable(Teg):
    global_uid = 0

    def __init__(self, name: str, value: Optional[float] = None):
        super(TegVariable, self).__init__(children=[])
        self.name = name
        self.value = value
        self.uid = TegVariable.global_uid
        TegVariable.global_uid += 1

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        if self.name == var_name:
            self.value = value


class TegConstant(TegVariable):

    def __init__(self, value: Optional[float], name: str = ''):
        super(TegConstant, self).__init__(name=name, value=value)

    def bind_variable(self, var_name: str, value: Optional[float]) -> None:
        pass


class TegAdd(Teg):
    name = 'add'
    operation = operator.add


class TegMul(Teg):
    name = 'mul'
    operation = operator.mul


class TegIntegral(Teg):

    def __init__(self, lower: TegVariable, upper: TegVariable, body: Teg, dvar: TegVariable):
        super(TegIntegral, self).__init__(children=[lower, upper, body])
        self.lower, self.upper, self.body = self.children
        self.dvar = dvar


class TegConditional(Teg):

    def __init__(self, var1: TegVariable, var2: TegVariable, if_body: Teg, else_body: Teg, allow_eq: bool = False):
        super(TegConditional, self).__init__(children=[var1, var2, if_body, else_body])
        self.var1, self.var2, self.if_body, self.else_body = self.children
        self.allow_eq = allow_eq


class TegTuple(Teg):

    def __init__(self, *args):
        super(TegTuple, self).__init__(children=args)


class TegLetIn(Teg):

    def __init__(self,
                 new_vars: TegTuple,
                 new_exprs: TegTuple,
                 expr: Teg):
        super(TegLetIn, self).__init__(children=[expr, *new_exprs])
        self.new_vars = new_vars
        self.new_exprs = self.children[1:]
        self.expr = self.children[0]


class TegFunction(Teg):

    def __init__(self, name, body, *args):
        super(TegLetIn, self).__init__(children=[body])
        self.name = name
        self.body = body
        self.args = args


class TegContext(dict):
    """Mapping from strings to TegVariables. """
    global_uid = 0

    def __init__(self, env: Optional[Dict[str, Any]] = None, parent: Optional['TegContext'] = None):
        self.uid = TegContext.global_uid
        TegContext.global_uid += 1
        super().update({} if env is None else env)
        self.parent = {} if parent is None else parent

    def __getitem__(self, key: str) -> Any:
        if key in self.env:
            return self.env[key]
        elif not self.parent:
            raise ValueError(f'No value for the variable "{key}" has been set.')
        else:
            return self.parent[key]
