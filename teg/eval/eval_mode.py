class EvalMode():
    def __init__(self, name=None):
        self.name = name
        pass

    def eval(self, *kwargs):
        raise NotImplementedError('This backend does not support evaluation')
