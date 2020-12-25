from teg.utils import overloads
from teg.ir.instr import (
    IR_Binary,
    IR_Variable,
    IR_Literal,
    IR_UnaryMath,
    IR_Assign,
    IR_Integrate,
    IR_Pack,
    IR_Call,
    IR_Function
)


@overloads(IR_Variable)
class PrintPass_Variable:
    def __str__(self):
        if hasattr(self, 'varname'):
            return self.varname
        else:
            return f'@{id(self)}{self._teg_var.name if self._teg_var is not None else ""}'


@overloads(IR_Literal)
class PrintPass_Literal:
    def __str__(self):
        return f'{self.value}'


@overloads(IR_Binary)
class PrintPass_Binary:
    def __str__(self):
        return f'{self.output} = {type(self).__name__}({self.input1}, {self.input2})'


@overloads(IR_UnaryMath)
class PrintPass_UnaryMath:
    def __str__(self):
        return f'{self.output} = {type(self).__name__}({self.inputs[0]})'


@overloads(IR_Assign)
class PrintPass_Assign:
    def __str__(self):
        return f'{self.output} = {self.inputs[0]}'


@overloads(IR_Pack)
class PrintPass_Pack:
    def __str__(self):
        return f"{self.output} = ({''.join([f'{inp},' for inp in self.inputs])})"


@overloads(IR_Integrate)
class PrintPass_Integrate:
    def __str__(self):
        full_str = f'Integrate {self.teg_var}: {self.lower_bound} -> {self.upper_bound}\n'
        full_str += str(self.call)
        return full_str


@overloads(IR_Call)
class PrintPass_Call:
    def __str__(self):
        return str(self.function)


@overloads(IR_Function)
class PrintPass_Function:
    def __str__(self):
        full_str = ''
        for instr in self.instrs:
            lines = str(instr).splitlines(keepends=False)
            for line in lines:
                full_str += '\t' + line + '\n'
        return full_str[:-1]
