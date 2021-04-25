from teg.math import (
    Sqr,
    Sqrt,
    Sin,
    Cos,
    ATan2,
    ASin
)

from .utils import overloads


class SimpleCMath():
    def simple_c(input_symbol, output_symbol, c_gen_fn, **kwargs):
        output_size = output_symbol.irtype().size

        if output_size == 1:
            return f'{output_symbol.__c_putstring__(**kwargs)} = ' \
                   f'{c_gen_fn(input_symbol.__c_getstring__(**kwargs))};'
        else:
            input_string = input_symbol.__c_getstring__(index='__iter__', broadcast=True, **kwargs)
            output_string = output_symbol.__c_putstring__(index="__iter__", broadcast=True, **kwargs)
            return f'for(int __iter__ = 0; __iter__ < {output_size}; __iter__++)\n' \
                   f'{output_string} = ' \
                   f'{c_gen_fn(input_string)};'


class MultivariateCMath():
    def multi_c(input_symbol, output_symbol, c_gen_fn, **kwargs):
        return f'{output_symbol.__c_putstring__(**kwargs)} = ' \
               f'{c_gen_fn(input_symbol)};'


@overloads(Sqr)
class CPass_Sqr(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'{x} * {x}', **kwargs)


@overloads(Sqrt)
class CPass_Sqrt(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'sqrt({x})', **kwargs)


@overloads(Sin)
class CPass_Sin(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'sin({x})', **kwargs)


@overloads(Cos)
class CPass_Cos(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'cos({x})', **kwargs)


@overloads(ASin)
class CPass_ASin(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'asin({x})', **kwargs)


@overloads(ATan2)
class CPass_ATan2(MultivariateCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        assert input_symbol.irtype().size == 2, 'atan2() takes exactly 2 arguments'
        return MultivariateCMath.multi_c(
                            input_symbol, output_symbol,
                            lambda symbol: (f'atan2({symbol.__c_getstring__(index=0, broadcast=False, **kwargs)},'
                                            f'{symbol.__c_getstring__(index=1, broadcast=False, **kwargs)})'
                                            ),
                            **kwargs
                            )
