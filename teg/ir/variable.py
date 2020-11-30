"""

"""
class Symbol:
    def __init__(self, name = None, ctype = None):
        self.name = name
        self.ctype = ctype
        pass

class Variable:
    def __init__(self, name = None, ctype = None):
        self.name = name
        self.ctype = ctype
        pass

def infer_const_type(v):
    if type(v) in [float, int]:
        return 'float'
    else:
        assert False, f'Couldn\'t match constant type'

class Constant:
    def __init__(self, name = None, val = None, ctype = None):
        self.name = name
        self.ctype = infer_const_type(val) if ctype else val
        self.val = val
        
        pass
