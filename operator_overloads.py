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


def overloads(to_cls):
    def overloaded(from_cls):
        """Dynamically inject all functions in `from_cls` into `to_cls`. """
        for func in filter(lambda x: callable(x), from_cls.__dict__.values()):
            setattr(to_cls, func.__name__, func)
    return overloaded


@overloads(Teg)
class TegOverloads:

    def __add__(self, other):
        return TegAdd([self, other])

    def __sub__(self, other):
        return self + (-other)

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
        self.sign *= -1
        return self


@overloads(TegVariable)
class TegVariableOverloads:

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{'-' if self.sign == -1 else ''}{self.name}{value}"

    def __repr__(self):
        return f'TegVariable(name={self.name}, value={self.value}, sign={self.sign})'

    def __neg__(self):
        return type(self)(name=self.name, value=self.value, sign=self.sign*-1)


@overloads(TegConstant)
class TegConstantOverloads:

    def __str__(self):
        return f'{"-" if self.sign == -1 else ""}{"" if not self.name else f"{self.name}="}{self.value}'

    def __repr__(self):
        return f'TegConstant(value={self.value}, name={self.name}, sign={self.sign})'


@overloads(TegIntegral)
class TegIntegralOverloads:

    def __str__(self):
        return f'int_{{{str(self.lower)}}}^{{{str(self.upper)}}} {str(self.body)} d{self.dvar.name}'


@overloads(TegConditional)
class TegConditionalOverloads:

    def __str__(self):
        return f'(({self.var1} <{"=" if self.allow_eq else ""} {self.var2}) ? {(self.if_body)} : {(self.else_body)})'


@overloads(TegTuple)
class TegTupleOverloads:

    def __str__(self):
        return ", ".join([str(e) for e in self.children])

    def __iter__(self):
        yield from (e for e in self.children)

    def __len__(self):
        return len(self.children)


@overloads(TegLetIn)
class TegLetInOverloads:

    def __str__(self):
        bindings = [f'{var}={expr}' for var, expr in zip(self.new_vars, self.new_exprs)]
        assignments = bindings[0] if len(bindings) == 1 else bindings
        return f'let {assignments} in {self.expr}'
