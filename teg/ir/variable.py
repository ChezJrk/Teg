"""

"""


class IR_Variable:
    def __init__(self, ir_type=None, label=None):
        # Generate name.
        self.name = 'a'
        pass

    def name(self):
        return self.name

    def ctype(self):
        return self.ctype


def infer_const_type(v):
    if type(v) in [float, int]:
        return 'float'
    else:
        assert False, f'Couldn\'t match constant type'


class IR_Literal:
    def __init__(self, val=None):
        self.val = val

    def value():
        return self.val
