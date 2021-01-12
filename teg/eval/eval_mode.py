class EvalMode():
    def __init__(self, name=None):
        self.name = name
        pass

    def eval(self, bindings, **kwargs):
        raise NotImplementedError('This backend does not support evaluation')
