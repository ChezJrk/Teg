"""Add operator overload methods to Teg classes. """
import inspect
import sys

from integrable_program import (
    Teg,
    TegConstant,
    TegVariable,
    TegAdd,
    TegMul,
    TegConditional,
    TegIntegral,
    TegTuple,
    TegLetIn,
)


class TegOverloads:
    overloads = Teg

    def __add__(self, other):
        return TegAdd([self, other])

    def __mul__(self, other):
        return TegMul([self, other])

    def __radd__(self, other):
        return other + self

    def __rmul__(self, other):
        return other * self

    def __pow__(self, exp):
        if exp == 0:
            return TegConstant(1)
        # TODO: this is naive linear, could be log. Also, should support more.
        assert isinstance(exp, int) and exp > 0, "We only support positive integer powers."
        return self * self**(exp - 1)

    def __str__(self):
        children = [str(child) for child in self.children]
        return f'{self.name}({", ".join(children)})'

    def __neg__(self):
        return type(self)(children=self.children, sign=self.sign*-1)


class TegVariableOverloads:
    overloads = TegVariable

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            other = TegConstant(other)
        return self.value < other.value

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{'-' if self.sign == -1 else ''}{self.name}{value}"

    def __repr__(self):
        return f'TegVariable(name={self.name}, value={self.value}, sign={self.sign})'

    def __neg__(self):
        return type(self)(name=self.name, value=self.value, sign=self.sign*-1)


class TegConstantOverloads:
    overloads = TegConstant

    def __str__(self):
        return f'{"-" if self.sign == -1 else ""}{"" if not self.name else f"{self.name}="}{self.value}'

    def __repr__(self):
        return f'TegConstant(value={self.value}, name={self.name}, sign={self.sign})'


class TegIntegralOverloads:
    overloads = TegIntegral

    def __str__(self):
        return f'int_{{{str(self.lower)}}}^{{{str(self.upper)}}} {str(self.body)} d{self.dvar.name}'


class TegConditionalOverloads:
    overloads = TegConditional

    def __str__(self):
        return f'if({self.var} < {self.const}): {self.if_body} else {self.else_body}'


class TegTupleOverloads:
    overloads = TegTuple

    def __str__(self):
        return ", ".join([str(e) for e in self.children])

    def __iter__(self):
        yield from (e for e in self.children)

    def __len__(self):
        return len(self.children)


class TegLetInOverloads:
    overloads = TegLetIn

    def __str__(self):
        bindings = [f'{var}={expr}' for var, expr in zip(self.new_vars, self.new_exprs)]
        assignments = bindings[0] if len(bindings) == 1 else bindings
        return f'let {assignments} in {self.var} = {self.expr}'


def inject_all_methods():

    def add_method(func, *tocls):
        for c in tocls:
            setattr(c, func.__name__, func)

    def filter_to_classes_in_module(cls):
        return inspect.isclass(cls) and 'operator_overloads.' in repr(cls)

    # Dynamically inject the body of the overloaded classes into Teg objects
    # by getting all classes in this module and injecting them into classes
    # specified by C.overloads.
    clsmembers = inspect.getmembers(sys.modules[__name__], filter_to_classes_in_module)
    for _, C in clsmembers:
        for func in filter(lambda x: callable(x), C.__dict__.values()):
            add_method(func, C.overloads)


inject_all_methods()
