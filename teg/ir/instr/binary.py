from .instr import IR_Binary


class IR_Add(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_Add, self).__init__(output=output, input1=input1, input2=input2)


class IR_Mul(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_Mul, self).__init__(output=output, input1=input1, input2=input2)


class IR_Divide(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_Divide, self).__init__(output=output, input1=input1, input2=input2)


class IR_LAnd(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_LAnd, self).__init__(output=output, input1=input1, input2=input2)


class IR_LOr(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_LOr, self).__init__(output=output, input1=input1, input2=input2)


class IR_Subtract(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_Subtract, self).__init__(output=output, input1=input1, input2=input2)


class IR_CompareLT(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_CompareLT, self).__init__(output=output, input1=input1, input2=input2)


class IR_CompareGT(IR_Binary):
    def __init__(self, output, input1, input2):
        super(IR_CompareGT, self).__init__(output=output, input1=input1, input2=input2)
