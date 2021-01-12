from enum import Enum
from typing import Any

from teg.utils import overloads
from teg.ir.instr import (
    IR_Binary,
    IR_IfElse,
    IR_CompareLT,
    IR_CompareGT,
    IR_Variable,
    IR_Literal,
    IR_UnaryMath,
    IR_Assign,
    IR_Integrate,
    IR_Pack,
    IR_Call,
    IR_Function
)


class Types(Enum):
    FLOAT = 1
    BOOL = 2


def ir_type_from(c: Any) -> Types:
    if type(c) in (float, int):
        return IR_Type(ctype=Types.FLOAT, size=1)
    elif type(c) is bool:
        return IR_Type(ctype=Types.BOOL, size=1)
    else:
        # Attempt to cast to float.
        try:
            float(c)
            return IR_Type(ctype=Types.FLOAT, size=1)
        except ValueError:
            assert False, f"Invalid literal type: {type(c)}"


class IR_Type:
    def __init__(self, ctype, size=1):
        self.ctype = ctype
        self.size = size

    def __eq__(self, o):
        return ((type(self) == type(o)) and
                (self.ctype == self.ctype) and
                (self.size == o.size))

    def __str__(self):
        STRINGS = {
            Types.FLOAT: 'float',
            Types.BOOL: 'bool'
        }
        return f'{STRINGS[self.ctype]}{f"[{self.size}]" if self.size > 1  else ""}'


def tegpass_typing(obj: Any, *args, **kwargs):
    assert '__tegpass_typing__' in dir(obj), 'Encountered unsupported object'
    out = obj.__tegpass_typing__(*args, **kwargs)

    return out


__DEFAULT_TYPE__ = IR_Type(Types.FLOAT, size=1)


@overloads(IR_Variable)
class TypingPass_Variable:

    def _try_infer_type_from_default(self):
        if not hasattr(self, '_irtype'):
            if self.default is None and self._teg_var is not None:
                self.default = self._teg_var.value  # Extract default from the linked tegvar.
            if self.default is not None:
                self.set_irtype(ir_type_from(self.default))
            elif self._teg_var is not None:
                self.set_irtype(__DEFAULT_TYPE__)

    def _combine_types(self, type1, type2):
        if type1 == type2 or type2 is None:
            return type1
        elif type1 is None:
            return type2

        if type1.ctype != type2.ctype:
            return None
        elif type1.size == 1:
            return type2
        elif type2.size == 1:
            return type1
        else:
            return None

    def set_irtype(self, irtype: IR_Type):
        if hasattr(self, '_irtype') and self._irtype is not None:
            assert self._combine_types(self._irtype, irtype) is not None,\
                   f'Conflicting type assignment {self._irtype} != {irtype}, for symbol {self.label}'
        else:
            self._irtype = None

        self._irtype = self._combine_types(self._irtype, irtype)

    def irtype(self) -> IR_Type:
        self._try_infer_type_from_default()
        if not hasattr(self, '_irtype'):
            return None
        return self._irtype


@overloads(IR_Literal)
class TypingPass_Literal:

    def _infer_type(self):
        self._irtype = ir_type_from(self.value)

    def set_irtype(self, irtype: IR_Type):
        raise ValueError("Can't set type for a literal")

    def irtype(self) -> IR_Type:
        self._infer_type()
        return self._irtype


@overloads(IR_Binary)
class TypingPass_Binary:

    def __tegpass_typing__(self):
        left_symbol, right_symbol = self.inputs
        left_type = left_symbol.irtype()
        right_type = right_symbol.irtype()

        assert left_type.ctype == right_type.ctype, 'Binary operands have incompatible types'
        assert ((left_type.size == right_type.size) or
                (left_type.size == 1) or (right_type.size == 1)), 'Binary operands have incompatible sizes'

        self.output.set_irtype(IR_Type(ctype=left_type.ctype, size=max(left_type.size, right_type.size)))


@overloads(IR_IfElse)
class TypingPass_IfElse:

    def __tegpass_typing__(self):
        tegpass_typing(self.if_call)
        tegpass_typing(self.else_call)

        assert self.if_call.output.irtype() == self.else_call.output.irtype(),\
               f'if and else branches have incompatible types:'\
               f'{self.if_call.output.irtype()} and {self.else_call.output.irtype()}'

        assert self.condition.irtype() == IR_Type(ctype=Types.BOOL, size=1),\
               f'Condition type must be a scalar boolean: {self.condition.irtype()}'

        self.output.set_irtype(self.if_call.output.irtype())


@overloads(IR_CompareGT)
class TypingPass_CompareGT:

    def __tegpass_typing__(self):
        left_symbol, right_symbol = self.inputs
        left_type = left_symbol.irtype()
        right_type = right_symbol.irtype()

        assert left_type.ctype == right_type.ctype, 'Binary operands have incompatible types'
        assert ((left_type.size == right_type.size) or
                (left_type.size == 1) or (right_type.size == 1)), 'Binary operands have incompatible sizes'

        self.output.set_irtype(IR_Type(ctype=Types.BOOL, size=left_type.size))


@overloads(IR_CompareLT)
class TypingPass_CompareLT:
    def __tegpass_typing__(self):
        left_symbol, right_symbol = self.inputs
        left_type = left_symbol.irtype()
        right_type = right_symbol.irtype()

        assert left_type.ctype == right_type.ctype, 'Binary operands have incompatible types'
        assert ((left_type.size == right_type.size) or
                (left_type.size == 1) or (right_type.size == 1)), 'Binary operands have incompatible sizes'

        self.output.set_irtype(IR_Type(ctype=Types.BOOL, size=left_type.size))


@overloads(IR_UnaryMath)
class TypingPass_UnaryMath:
    def __tegpass_typing__(self):
        input_symbol = self.inputs[0]
        input_type = input_symbol.irtype()

        if hasattr(self.fn_class, 'output_size'):
            output_size = self.fn_class.output_size(input_type.size)
        else:
            output_size = input_type.size
        self.output.set_irtype(IR_Type(ctype=input_type.ctype, size=output_size))


@overloads(IR_Assign)
class TypingPass_Assign:
    def __tegpass_typing__(self):
        input_symbol = self.inputs[0]
        input_type = input_symbol.irtype()
        self.output.set_irtype(IR_Type(ctype=input_type.ctype, size=input_type.size))


@overloads(IR_Pack)
class TypingPass_Pack:
    def __tegpass_typing__(self):
        input_symbols = self.inputs
        input_type = input_symbols[0].irtype()

        assert input_type.size == 1,\
               f'Teg IR currently does not support multidimensional packing {input_symbols[0]}. '\
               f'Use compatibility mode (backend="numpy") instead.'
        for input_symbol in input_symbols:
            assert input_symbol.irtype() == input_type, 'All tuple elements must have the same type'

        self.output.set_irtype(IR_Type(ctype=input_type.ctype, size=len(self.inputs)))


@overloads(IR_Integrate)
class TypingPass_Integrate:
    def __tegpass_typing__(self):
        assert self.upper_bound.irtype() == self.lower_bound.irtype(),\
               f'Bounds are of different types: {self.upper_bound.irtype()} and {self.lower_bound.irtype()}'
        self.teg_var.set_irtype(self.upper_bound.irtype())

        tegpass_typing(self.call)
        self.output.set_irtype(self.call.output.irtype())


@overloads(IR_Call)
class TypingPass_Call:
    def __tegpass_typing__(self):
        for fn_symbol, call_symbol in zip(self.function.inputs, self.inputs):
            fn_symbol.set_irtype(call_symbol.irtype())
        tegpass_typing(self.function)

        if not isinstance(self.output, IR_Literal):
            self.output.set_irtype(self.function.output.irtype())


@overloads(IR_Function)
class TypingPass_Function:
    def __tegpass_typing__(self):
        for instr in self.instrs:
            tegpass_typing(instr)


def infer_typing(ir_func: IR_Function):
    tegpass_typing(ir_func)
