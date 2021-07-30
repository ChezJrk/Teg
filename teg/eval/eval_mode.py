class EvalMode():
    def __init__(self, name=None):
        self.name = name
        pass

    def eval(self, bindings, **kwargs):
        raise NotImplementedError('This backend does not support evaluation')

    @staticmethod
    def assert_is_available(**kwargs):
        raise AssertionError('This backend is not available')
