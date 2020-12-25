from .instr import IR_Instruction, IR_Call
from .variable import IR_Variable, IR_Symbol, IR_Literal


class IR_Integrate(IR_Instruction):
    def __init__(self, output: IR_Variable, call: IR_Call,
                 teg_var: IR_Variable, lower_bound: IR_Symbol, upper_bound: IR_Symbol):
        super(IR_Integrate, self).__init__(output=output,
                                           inputs=set([inp for inp in call.inputs if inp is not teg_var]) |
                                           (set([upper_bound]) if not isinstance(upper_bound, IR_Literal) else set()) |
                                           (set([lower_bound]) if not isinstance(lower_bound, IR_Literal) else set()))
        self.call = call
        self.teg_var = teg_var
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
