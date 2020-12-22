def overloads(to_cls):
    def overloaded(from_cls):
        """Dynamically inject all functions in `from_cls` into `to_cls`. """
        for func in filter(lambda x: callable(x), from_cls.__dict__.values()):
            setattr(to_cls, func.__name__, func)
    return overloaded
