"""Add operator overload methods to Teg classes. """
import numpy as np

from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    Cond,
    Teg,
    Tup,
    LetIn,
    try_making_teg_const,
)


def overloads(to_cls):
    def overloaded(from_cls):
        """Dynamically inject all functions in `from_cls` into `to_cls`. """
        for func in filter(lambda x: callable(x), from_cls.__dict__.values()):
            setattr(to_cls, func.__name__, func)
    return overloaded


@overloads(ITeg)
class TegOverloads:

    def __add__(self, other):
        return Add([self, try_making_teg_const(other)])

    def __sub__(self, other):
        return self + (-try_making_teg_const(other))

    def __mul__(self, other):
        return Mul([self, try_making_teg_const(other)])

    def __radd__(self, other):
        return try_making_teg_const(other) + self

    def __rmul__(self, other):
        return try_making_teg_const(other) * self

    def __pow__(self, exp):
        exp = try_making_teg_const(exp)
        if exp.value == 0:
            return Const(1)
        # TODO: this is naive linear, could be log. Also, should support more.
        assert isinstance(exp.value, int) and exp.value > 0, "We only support positive integer powers."
        return self * self**(exp.value - 1)

    def __str__(self):
        children = [str(child) for child in self.children]
        return f'{self.name}({", ".join(children)})'

    def __neg__(self):
        return Const(-1) * self

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children))


@overloads(Var)
class TegVariableOverloads:

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.name == other.name
                and self.uid == other.uid)

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{self.name}{value}"

    def __repr__(self):
        return f'TegVariable(name={self.name}, value={self.value})'


@overloads(Const)
class TegConstantOverloads:

    def __str__(self):
        return f'{"" if not self.name else f"{self.name}="}{self.value}'

    def __repr__(self):
        return f'TegConstant(value={self.value}, name={self.name})'

    def __eq__(self, other):
        return self.value == other.value


@overloads(Add)
class TegAddOverloads:

    def __str__(self):
        return f'({self.children[0]} + {self.children[1]})'


@overloads(Mul)
class TegMulOverloads:

    def __str__(self):
        return f'({self.children[0]} * {self.children[1]})'


@overloads(Teg)
class TegIntegralOverloads:

    def __str__(self):
        return f'(int_{{{self.dvar.name}=[{str(self.lower)}, {str(self.upper)}]}} {str(self.body)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.dvar == other.dvar)


@overloads(Cond)
class TegConditionalOverloads:

    def __str__(self):
        return f'(({self.lt_expr} <{"=" if self.allow_eq else ""} 0) ? {(self.if_body)} : {(self.else_body)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.allow_eq == other.allow_eq)


@overloads(Tup)
class TegTupleOverloads:

    def __str__(self):
        return ", ".join([str(e) for e in self.children])

    def __iter__(self):
        yield from (e for e in self.children)

    def __len__(self):
        return len(self.children)


@overloads(LetIn)
class TegLetInOverloads:

    def __str__(self):
        bindings = [f'{var}={expr}' for var, expr in zip(self.new_vars, self.new_exprs)]
        if len(bindings) == 1:
            assignments = bindings[0]
        else:
            joined_assignments = ',\n\t'.join(bindings)
            assignments = f'[{joined_assignments}]'
        return f'let {assignments} in {self.expr}'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.new_vars == other.new_vars)
