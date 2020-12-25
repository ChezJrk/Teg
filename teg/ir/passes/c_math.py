from teg.math import (
    Sqr,
    Sqrt
)

from .utils import overloads


class SimpleCMath():
    def simple_c(input_symbol, output_symbol, c_gen_fn, **kwargs):
        input_size = input_symbol.irtype().size

        if input_size == 1:
            return f'{output_symbol.__c_putstring__(**kwargs)} = ' \
                   f'{c_gen_fn(input_symbol.__c_getstring__(**kwargs))};'
        else:
            input_string = input_symbol.__c_getstring__(index='__iter__', broadcast=True, **kwargs)
            output_string = output_symbol.__c_putstring__(index="__iter__", broadcast=True, **kwargs)
            return f'for(int __iter__ = 0; __iter__ < {input_size}; __iter__++)\n' \
                   f'{output_string} = ' \
                   f'{c_gen_fn(input_string)};'


@overloads(Sqr)
class CPass_Sqr(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'{x} * {x}', **kwargs)


@overloads(Sqrt)
class CPass_Sqrt(SimpleCMath):
    def to_c(input_symbol, output_symbol, **kwargs):
        return SimpleCMath.simple_c(input_symbol, output_symbol, lambda x: f'sqrt({x})', **kwargs)
