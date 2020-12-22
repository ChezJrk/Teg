from .instr import IR_Instruction


class IR_IfElse(IR_Instruction):
    def __init__(self, output, condition, if_call, else_call):
        super(IR_IfElse, self).__init__(output=output, inputs=[*if_call.inputs, *else_call.inputs, condition])
        self.if_call = if_call
        self.else_call = else_call
        self.condition = condition
