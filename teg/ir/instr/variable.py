"""

"""


class IR_Symbol():
    def __init__(self):
        pass


class IR_Variable(IR_Symbol):
    def __init__(self, label=None, default=None, var=None):
        self.name = None
        self.default = default
        self.label = label if label is not None else '_t'

        # Book-keeping
        self._teg_var = var
        pass
    
    def __str__(self):
        return f'IR_Variable{{label={self.label}, default={self.default}, var={self._teg_var}}}'


def infer_const_type(v):
    if type(v) in [float, int]:
        return 'float'
    else:
        assert False, 'Couldn\'t match constant type'


class IR_Literal(IR_Symbol):
    def __init__(self, value=None):
        self.value = value

    def value(self):
        return self.value
