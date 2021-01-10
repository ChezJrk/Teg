class IR_Instruction:
    def __init__(self, output, inputs):
        self.output = output
        self.inputs = inputs


class IR_Binary(IR_Instruction):
    def __init__(self, output, input1, input2):
        super(IR_Binary, self).__init__(output=output, inputs=[input1, input2])
        self.input1 = input1
        self.input2 = input2


class IR_UnaryMath(IR_Instruction):
    def __init__(self, output, input, fn_class):
        super(IR_UnaryMath, self).__init__(output=output, inputs=[input])
        self.fn_class = fn_class


class IR_Call(IR_Instruction):
    def __init__(self, output, inputs, function):
        super(IR_Call, self).__init__(output=output, inputs=inputs)
        self.function = function


class IR_Function():
    def __init__(self, instrs, output, inputs, label=None):
        self.instrs = instrs
        self.output = output
        self.inputs = inputs
        self.label = label

    def add_instr(self, instr: IR_Instruction):
        self.instrs.append(instr)


class IR_Assign(IR_Instruction):
    def __init__(self, output, input):
        super(IR_Assign, self).__init__(output=output, inputs=[input])


class IR_Pack(IR_Instruction):
    def __init__(self, output, inputs):
        super(IR_Pack, self).__init__(output=output, inputs=inputs)
