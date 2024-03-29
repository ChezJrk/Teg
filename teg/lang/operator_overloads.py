"""Add operator overload methods to Teg classes. """
from .base import (
    SmoothFunc,
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Tup,
    LetIn,
    Invert,
    try_making_teg_const,
    ITegBool,
    Bool,
    And,
    Or,
)
from .markers import Placeholder
from .teg import Teg, TegVar
from .extended import BiMap, Delta
from teg.utils import overloads


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

    def __rsub__(self, other):
        return try_making_teg_const(other) - self

    def __rmul__(self, other):
        return try_making_teg_const(other) * self

    def __rtruediv__(self, other):
        return try_making_teg_const(other) / self

    def __pow__(self, exp):
        exp = try_making_teg_const(exp)
        if exp.value == 0:
            return Const(1)
        # NOTE: this is naive linear, could be log. Also, should support more.
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

    def __le__(self, other):
        return Bool(self, other, allow_eq=True)

    def __gt__(self, other):
        return Bool(other, self)

    def __ge__(self, other):
        return Bool(other, self, allow_eq=True)

    def __contains__(self, item):
        return (item == self) or any([item in child for child in self.children])

    def __hash__(self):
        if hasattr(self, 'hash_cache'):
            return self.hash_cache
        self.hash_cache = hash(tuple(hash(e) for e in self.children))
        return self.hash_cache


@overloads(Var)
class TegVariableOverloads:

    def __eq__(self, other):
        return (isinstance(self, Var) and isinstance(other, Var)
                and self.name == other.name
                and self.uid == other.uid)

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{self.name}({self.uid}){value}"

    def __repr__(self):
        value = '' if self.value is None else f', value={self.value}'
        return f'Var(name={self.name}{value})'

    def __ete__(self):
        return f'Var_{self.name}:{self.value}'

    def __hash__(self):
        return f'{self.name}_{self.uid}'.__hash__()

    def __copy__(self):
        return Var(name=self.name,
                   uid=self.uid,
                   value=self.value)


@overloads(TegVar)
class TegTegVariableOverloads:

    def __str__(self):
        value = '' if self.value is None else f'={self.value}'
        return f"{self.name}({self.uid}){value}"

    def __repr__(self):
        value = '' if self.value is None else f', value={self.value}'
        return f'TegVar(name={self.name}{value}, uid={self.uid})'

    def __copy__(self):
        e = TegVar(name=self.name,
                   uid=self.uid)
        e.__setattr__('value', self.value)
        return e


@overloads(Placeholder)
class TegPlaceholderOverloads:

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.signature == other.signature)

    def __str__(self):
        return f"{self.signature}"

    def __repr__(self):
        return f'Placeholder(name={self.name}, signature={self.signature})'

    def __copy__(self):
        e = Placeholder(name=self.name,
                        signature=self.signature)
        e.__setattr__('uid', self.uid)
        return e


@overloads(Const)
class TegConstantOverloads:

    def __str__(self):
        return f'{"" if not self.name else f"{self.name}="}{self.value:1.3f}'

    def __repr__(self):
        name = '' if self.name == '' else f', name={self.name}'
        return f'Const(value={self.value}{name})'

    def __eq__(self, other):
        return (type(self) == type(other)) and self.value == other.value

    def __copy__(self):
        e = Const(name=self.name,
                  value=self.value)
        e.__setattr__('uid', self.uid)
        return e


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


@overloads(SmoothFunc)
class SmoothFuncOverloads:

    def __str__(self):
        return f'{self.name}({self.expr})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and self.expr == other.expr)

    def __repr__(self):
        return f'{self.name}({repr(self.expr)})'


@overloads(Tup)
class TegTupleOverloads:

    def __str__(self):
        return "(" + ", ".join([str(e) for e in self.children]) + ")"

    def __repr__(self):
        return f'Tup({[repr(e) for e in self.children]})'

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


@overloads(BiMap)
class BiMapOverloads:

    def __str__(self):
        bindings = [f'{var}->{expr}' for var, expr in zip(self.targets, self.target_exprs)]
        if self.source_exprs:
            inv_bindings = [f'{var}->{expr}' for var, expr in zip(self.sources, self.source_exprs)]
        else:
            inv_bindings = ['no-inverse']

        if len(bindings) == 1:
            assignments = bindings[0]
        else:
            joined_assignments = ',\n\t'.join(bindings)
            assignments = f'[{joined_assignments}]'

        if len(inv_bindings) == 1:
            inv_assignments = inv_bindings[0]
        else:
            inv_joined_assignments = ',\n\t'.join(inv_bindings)
            inv_assignments = f'[{inv_joined_assignments}]'

        inv_jacobian = f'invjac = {self.inv_jacobian}' if self.inv_jacobian else 'no-inv_jacobian'
        target_upper_bounds = f'ubs = {self.target_upper_bounds}' if self.inv_jacobian else 'no-upper-bounds'
        target_lower_bounds = f'lbs = {self.target_lower_bounds}' if self.inv_jacobian else 'no-lower-bounds'

        return (f'map({id(self)}) {assignments} {inv_assignments} in {self.expr}')
                # f' with {inv_jacobian},'
                # f' {target_lower_bounds}, {target_upper_bounds}')

    def __repr__(self):
        return f'BiMap({repr(self.expr)}, {repr(self.targets)}, {repr(self.target_exprs)}, {repr(self.sources)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and all(e1 == e2 for e1, e2 in zip(self.children, other.children))
                and self.sources == other.sources
                and self.targets == other.targets)


@overloads(Delta)
class DeltaOverloads:

    def __str__(self):
        delta_symbol = '\u03B4'
        return f'{delta_symbol}({self.expr})'

    def __repr__(self):
        return f'Delta({repr(self.expr)})'

    def __eq__(self, other):
        return (type(self) == type(other)
                and len(self.children) == len(other.children)
                and self.expr == other.expr)


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
