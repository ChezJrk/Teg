from .base import Var


class Placeholder(Var):
    """Replaceable tag. """
    def __init__(self, name: str = '', signature: str = ''):
        super(Placeholder, self).__init__(name=name)
        self.signature = signature
