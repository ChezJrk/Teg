"""Add operator overload methods to Teg classes. """
import numpy as np

from integrable_program import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    Tup,
    LetIn,
    Invert,
    try_making_teg_const,
    ITegBool,
    Bool,
    And,
    Or,
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

    def __truediv__(self, other):
        return Mul([self, Invert(try_making_teg_const(other))])

    def __radd__(self, other):
        return try_making_teg_const(other) + self

    def __rmul__(self, other):
        return try_making_teg_const(other) * self

    def __rtruediv__(self, other):
        return try_making_teg_const(other) / self

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

    def __repr__(self):
        children = [repr(child) for child in self.children]
        return f'{self.name}({", ".join(children)})'

    def __neg__(self):
        return Const(-1) * self

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children))

    def __lt__(self, other):
        return Bool(self, other)

    def __leq__(self, other):
        return Bool(self, other, allow_eq=True)

    def __gt__(self, other):
        return Bool(other, self)

    def __geq__(self, other):
        return Bool(other, self, allow_eq=True)


@overloads(Var)
class TegVariableOverloads:

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.name == other.name
                and self.uid == other.uid)

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{self.name}{value}"

    def __repr__(self):
        value = '' if self.value is None else f', value={self.value}'
        return f'Var(name={self.name}{value})'


@overloads(Const)
class TegConstantOverloads:

    def __str__(self):
        return f'{"" if not self.name else f"{self.name}="}{self.value}'

    def __repr__(self):
        name = '' if self.name == '' else f', name={self.name}'
        return f'Const(value={self.value}{name})'

    def __eq__(self, other):
        return self.value == other.value


@overloads(Add)
class TegAddOverloads:

    def __str__(self):
        return f'({self.children[0]} + {self.children[1]})'

    def __repr__(self):
        return f'Add({repr(self.children[0])}, {repr(self.children[1])})'


@overloads(Mul)
class TegMulOverloads:

    def __str__(self):
        return f'({self.children[0]} * {self.children[1]})'

    def __repr__(self):
        return f'Mul({repr(self.children[0])}, {repr(self.children[1])})'


@overloads(Invert)
class InvertOverloads:

    def __str__(self):
        return f'(1 / {self.child})'

    def __repr__(self):
        return f'Invert({repr(self.child)})'


@overloads(Teg)
class TegIntegralOverloads:

    def __str__(self):
        return f'(int_{{{self.dvar.name}=[{self.lower}, {self.upper}]}} {self.body})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.dvar == other.dvar)

    def __repr__(self):
        return f'Teg({repr(self.lower)}, {repr(self.upper)}, {repr(self.body)}, {repr(self.dvar)})'


@overloads(IfElse)
class TegConditionalOverloads:

    def __str__(self):
        return f'(({self.cond}) ? {(self.if_body)} : {(self.else_body)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.cond == other.cond)

    def __repr__(self):
        return f'IfElse({repr(self.cond)}, {repr(self.if_body)}, {repr(self.else_body)})'


@overloads(Tup)
class TegTupleOverloads:

    def __str__(self):
        return ", ".join([str(e) for e in self.children])

    def __repr__(self):
        return f'Tup([{repr(e) for e in self.children}])'

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

    def __repr__(self):
        return f'LetIn({repr(self.new_vars)}, {repr(self.new_exprs)}, {repr(self.expr)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and sum(e1 == e2 for e1, e2 in zip(self.children, other.children)) == len(self.children)
                and self.new_vars == other.new_vars)


@overloads(ITegBool)
class ITegBoolOverloads:

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __eq__(self, other):
        return type(self) == type(other) and self.left_expr == other.left_expr and self.right_expr == other.right_expr


@overloads(Bool)
class BoolOverloads:

    def __str__(self):
        return f'{self.left_expr} < {self.right_expr}'

    def __repr__(self):
        return f'Bool({repr(self.left_expr)}, {repr(self.right_expr)})'


@overloads(And)
class AndOverloads:

    def __str__(self):
        return f'({self.left_expr} & {self.right_expr})'

    def __repr__(self):
        return f'And({repr(self.left_expr)}, {repr(self.right_expr)})'


@overloads(Or)
class OrOverloads:

    def __str__(self):
        return f'({self.left_expr} | {self.right_expr})'

    def __repr__(self):
        return f'Or({repr(self.left_expr)}, {repr(self.right_expr)})'
