from typing import Any
from teg.utils import overloads

from teg.ir.instr import (
    IR_Instruction,
    IR_Variable,
    IR_Literal,
    IR_Assign,

    IR_Binary,
    IR_CompareLT,
    IR_CompareGT,
    IR_LAnd,
    IR_LOr,
    IR_Add,
    IR_Mul,
    IR_Divide,

    IR_UnaryMath,
    IR_Pack,
    IR_Integrate,
    IR_IfElse,

    IR_Call,
    IR_Function
)

from .typing import infer_typing
from .typing import Types

from teg import __data_path__

from . import c_math


def indent(code, times=1):
    if not code:
        return code
    lines = code.splitlines(keepends=False)
    return ''.join(['\t' * times + line + '\n' for line in lines])[:-1]


def typename_from_irtype(irtype, float_width='float'):
    if irtype.ctype == Types.FLOAT:
        return float_width
    elif irtype.ctype == Types.BOOL:
        return 'bool'


def tegpass_c(obj: Any, *args, **kwargs):
    assert '__tegpass_c__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    a, b = obj.__tegpass_c__(*args, **kwargs)

    if 'None' in a:
        print(a)
    assert 'None' not in a, f'STOP {type(obj)}'
    return a, b


def c_symbolstring(obj: Any, *args, **kwargs):
    assert '__c_symbolstring__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_symbolstring__(*args, **kwargs)


def c_declstring(obj: Any, *args, **kwargs):
    assert '__c_declstring__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_declstring__(*args, **kwargs)


def c_typestring(obj: Any, *args, **kwargs):
    assert '__c_typestring__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_typestring__(*args, **kwargs)


def c_getstring(obj: Any, *args, **kwargs):
    assert '__c_getstring__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_getstring__(*args, **kwargs)


def c_putstring(obj: Any, *args, **kwargs):
    assert '__c_putstring__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_putstring__(*args, **kwargs)


def c_gather_symbols(obj: Any, *args, **kwargs):
    assert '__c_gather_symbols__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_gather_symbols__(*args, **kwargs)


def c_gather_functions(obj: Any, *args, **kwargs):
    assert '__c_gather_functions__' in dir(obj), f'Encountered unsupported object {type(obj)}'
    return obj.__c_gather_functions__(*args, **kwargs)


def generate_unique_name(name_ctx, label=None):
    if label not in name_ctx:
        name_ctx[label] = 1

    multiplicity = name_ctx[label]
    name_ctx[label] += 1

    return f'{label}_{multiplicity}'


class CTemplateFunction:
    def __init__(self, template_code, substitutions, name=None):
        self.template_code = template_code
        self.substitutions = substitutions
        self.name = name

    def __tegpass_c__(self, name_ctx, **kwargs):
        fn = self.template_code
        for find_str, replace_str in self.substitutions.items():
            fn = fn.replace(find_str, replace_str)

        return fn, []

    def __c_symbolstring__(self, name_ctx, **kwargs):
        assert self.name is not None, 'Function has no symbol name'
        return self.name


@overloads(IR_Instruction)
class CPass_Instruction:

    def __tegpass_c__(self, name_ctx, **kwargs):
        raise NotImplementedError(f'C backend does not support instruction class "{type(self)}"')

    def __c_gather_symbols__(self):
        raise NotImplementedError(f'C backend does not support instruction class "{type(self)}"')


@overloads(IR_Variable)
class CPass_Variable:

    def __resolve_name__(self, name_ctx):
        if not hasattr(self, 'varname'):
            self.varname = generate_unique_name(name_ctx, label=self.label)

    def __resolve_ctype_mode__(self):
        if not hasattr(self, '_ctype_mode'):
            self._ctype_mode = 'native'

    def __resolve_c_typename__(self, **kwargs):
        if not hasattr(self, '_c_typename'):
            c_options = {
                'float_width': kwargs.get('float_width', 'float')
            }
            self._c_typename = typename_from_irtype(self.irtype(), **c_options)

    def __c_declstring__(self, name_ctx, **kwargs):
        self.__resolve_name__(name_ctx)
        self.__resolve_ctype_mode__()
        self.__resolve_c_typename__(**kwargs)

        assert self.irtype() is not None,\
               f'Unable to infer type for symbol "{self.__c_str__()}"'

        if self._ctype_mode == 'native':
            return f'{self._c_typename} {self.varname}' if self.irtype().size == 1 else \
               f'generic_array<{self._c_typename},{self.irtype().size}> {self.varname}'
        elif self._ctype_mode == 'stl':
            return f'{self._c_typename} {self.varname}' if self.irtype().size == 1 else \
               f'std::array<{self._c_typename},{self.irtype().size}> {self.varname}'
        elif self._ctype_mode == 'py':
            return f'{self._c_typename} {self.varname}' if self.irtype().size == 1 else \
               f'pybind11::list {self.varname}({self.irtype().size})'

    def __c_typestring__(self, name_ctx, **kwargs):
        self.__resolve_ctype_mode__()
        self.__resolve_c_typename__(**kwargs)
        assert self.irtype() is not None,\
               f'Unable to infer type for symbol "{self.__c_str__()}"'

        if self._ctype_mode == 'native':
            return f'{self._c_typename}' if self.irtype().size == 1 else \
               f'generic_array<{self._c_typename},{self.irtype().size}>'
        elif self._ctype_mode == 'stl':
            return f'{self._c_typename}' if self.irtype().size == 1 else \
               f'std::array<{self._c_typename},{self.irtype().size}>'
        elif self._ctype_mode == 'py':
            return f'{self._c_typename}' if self.irtype().size == 1 else \
               'pybind11::list'

    def __c_symbolstring__(self, name_ctx, **kwargs):
        self.__resolve_name__(name_ctx)
        return f'{self.varname}'

    def __c_putstring__(self, name_ctx, index=None, broadcast=False, **kwargs):
        self.__resolve_ctype_mode__()
        varname = self.__c_symbolstring__(name_ctx=name_ctx)

        if self.irtype().size == 1:
            assert index is None or broadcast
            return f'{varname}'
        else:
            assert index is not None
            return f'{varname}[{index}]'

    def __c_getstring__(self, name_ctx, index=None, broadcast=False, **kwargs):
        return self.__c_putstring__(name_ctx=name_ctx, index=index, broadcast=broadcast, **kwargs)

    def __c_gather_symbols__(self):
        yield self

    def __c_str__(self):
        superstr = str(self)
        return f"{superstr}, C-Name: '{self.varname}'"

    def set_ctype_mode(self, _ctype_mode):
        self._ctype_mode = _ctype_mode

    def ctype_mode(self):
        self.__resolve_ctype_mode__()
        return self._ctype_mode

    def __eq__(self, o):
        return self is o


@overloads(IR_Literal)
class CPass_Literal:

    def __resolve_c_typename__(self, **kwargs):
        if not hasattr(self, '_c_typename'):
            c_options = {
                'float_width': kwargs.get('float_width', 'float')
            }
            self._c_typename = typename_from_irtype(self.irtype(), **c_options)

    def __tegpass_c__(self, name_ctx, **kwargs):
        self.__resolve_c_typename__(**kwargs)
        return f'{self.value}f' if self._c_typename == 'float' else f'{self.value}',\
               []

    def __c_declstring__(self, name_ctx, **kwargs):
        return None

    def __c_typestring__(self, name_ctx, **kwargs):
        self.__resolve_c_typename__(**kwargs)
        assert self.irtype().size == 1, 'No C support for array literals'
        return f'{self._c_typename}' if self.irtype().size == 1 else \
               f'{self._c_typename}[{self.irtype().size}]'

    def __c_symbolstring__(self, name_ctx, **kwargs):
        self.__resolve_c_typename__(**kwargs)
        return (max((f'{self.value:.1f}f', f'{self.value}f'), key=len) if self._c_typename == 'float'
                else f'{self.value}')

    def __c_putstring__(self, name_ctx, index=None, broadcast=False, **kwargs):
        raise NotImplementedError('Cannot assign to a literal')

    def __c_getstring__(self, name_ctx, index=None, broadcast=False, **kwargs):
        return self.__c_symbolstring__(name_ctx=name_ctx, **kwargs)

    def __c_gather_symbols__(self):
        pass


@overloads(IR_Binary)
class CPass_Binary:

    def __tegpass_c_binary__(self, name_ctx, op='+', **kwargs):
        left_symbol, right_symbol = self.inputs
        left_size = left_symbol.irtype().size
        right_size = right_symbol.irtype().size

        output_symbol = self.output

        if left_size == 1 and right_size == 1:
            return f'{c_symbolstring(output_symbol, name_ctx, **kwargs)} = '\
                   f'{c_symbolstring(left_symbol, name_ctx, **kwargs)}'\
                   f' {op} '\
                   f'{c_symbolstring(right_symbol, name_ctx, **kwargs)};',\
                   []
        else:
            for_size = left_size if right_size == 1 else right_size
            left_idx = c_getstring(left_symbol, name_ctx, '__iter__', True, **kwargs)
            right_idx = c_getstring(right_symbol, name_ctx, '__iter__', True, **kwargs)
            return f'for(int __iter__ = 0; __iter__ < {for_size}; __iter__++)' \
                   f'   {c_putstring(output_symbol, name_ctx, "__iter__", False, **kwargs)} = '\
                   f'{left_idx} {op} {right_idx};',\
                   []

    def __tegpass_c__(self, name_ctx, **kwargs):
        raise NotImplementedError(f'C backend does not support binary instruction "{type(self)}"')

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


@overloads(IR_CompareGT)
class CPass_CompareGT:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='>', **kwargs)


@overloads(IR_CompareLT)
class CPass_CompareLT:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='<', **kwargs)


@overloads(IR_Add)
class CPass_Add:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='+', **kwargs)


@overloads(IR_Mul)
class CPass_Mul:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='*', **kwargs)


@overloads(IR_Divide)
class CPass_Divide:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='/', **kwargs)


@overloads(IR_LOr)
class CPass_LOr:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='||', **kwargs)


@overloads(IR_LAnd)
class CPass_LAnd:
    def __tegpass_c__(self, name_ctx, **kwargs):
        return self.__tegpass_c_binary__(name_ctx, op='&&', **kwargs)


@overloads(IR_UnaryMath)
class CPass_UnaryMath:
    def __tegpass_c__(self, name_ctx, **kwargs):
        assert len(self.inputs) == 1
        assert hasattr(self.fn_class, 'to_c'),\
               f'Math operation {self.fn_class} has no defined C transformation. '\
               f'Overload the to_c() method to define a transformation.'\
               f'See teg/passes/c_math.py for examples'
        return self.fn_class.to_c(self.inputs[0], self.output, name_ctx=name_ctx, **kwargs), []

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


@overloads(IR_Assign)
class CPass_Assign:
    def __tegpass_c__(self, name_ctx, **kwargs):
        assert len(self.inputs) <= 1, 'Can\'t assign multiple inputs to an output'
        assert len(self.inputs) > 0, 'No inputs in assign statement'

        input_symbol = self.inputs[0]
        input_size = input_symbol.irtype().size
        output_symbol = self.output
        output_size = output_symbol.irtype().size

        if input_size > 1 or output_size > 1:
            return f'for(int __iter__ = 0; __iter__ < {input_size}; __iter__++)\n' \
                   f'{c_putstring(output_symbol, name_ctx, "__iter__", False, **kwargs)} = ' \
                   f'{c_getstring(input_symbol, name_ctx, "__iter__", False, **kwargs)};', []
        else:
            return f'{c_putstring(output_symbol, name_ctx, **kwargs)} = ' \
                   f'{c_getstring(input_symbol, name_ctx, **kwargs)};', []

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


@overloads(IR_Pack)
class CPass_Pack:
    def __tegpass_c__(self, name_ctx, **kwargs):
        assert len(self.inputs) > 1, 'Attempting to pack a single variable into a tuple'
        input_symbols = self.inputs
        output_symbol = self.output

        c_string = ''
        for idx, input_symbol in enumerate(input_symbols):
            c_string += f'{c_putstring(output_symbol, name_ctx, f"{idx}", False, **kwargs)} ='\
                        f'{c_getstring(input_symbol, name_ctx, **kwargs)};\n'

        return c_string[:-1], []

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


_INTEGRATOR_MAP = {
        'trapezoidal_quadrature': 'trapezoid_rule.template.c',
        'rectangular_quadrature': 'rectangular_rule.template.c',
        'monte_carlo_uniform': 'monte_carlo.template.c'
    }


@overloads(IR_Integrate)
class CPass_Integrate:
    def __tegpass_c__(self, name_ctx, num_samples=50, **kwargs):
        # print('CALL: ', type(self.call.output))
        fn_call_code, fn_call_methods = tegpass_c(self.call, name_ctx, **kwargs)

        is_code_device = kwargs.get('device_code')

        if 'integration_mode' not in dir(self) or self.integration_mode is None:
            self.integration_mode = 'rectangular_quadrature'

        integrator_fn_name = generate_unique_name(name_ctx,
                                                  label=f'{kwargs.get("fn_name_prefix", "")}integrator')

        integrator_template_file = _INTEGRATOR_MAP[self.integration_mode]
        _template = open(__data_path__ + '/templates/' + integrator_template_file, 'r')
        template = _template.read()
        _template.close()

        input_symbols = self.inputs

        input_declstrings = [c_declstring(symbol, name_ctx, **kwargs) + ','
                             for symbol in input_symbols
                             if c_declstring(symbol, name_ctx, **kwargs)]
        call_string = ''.join(input_declstrings)

        if call_string and call_string[-1] == ',':
            call_string = call_string[:-1]

        if fn_call_code:
            fn_call_code = f'{c_declstring(self.call.output, name_ctx, **kwargs)};\n\t\t{fn_call_code};'

        integrator_function = CTemplateFunction(
                                            template,
                                            {
                                                r'%TEG_VAR%': c_symbolstring(self.teg_var, name_ctx, **kwargs),
                                                r'%TEG_VAR_TYPE%': c_typestring(self.teg_var, name_ctx, **kwargs),
                                                r'%UPPER_VAR%': c_symbolstring(self.upper_bound, name_ctx, **kwargs),
                                                r'%LOWER_VAR%': c_symbolstring(self.lower_bound, name_ctx, **kwargs),
                                                r'%CALL_FN%': fn_call_code,
                                                r'%CALL_LIST%': call_string,
                                                r'%CALL_OUT_VAR%': c_symbolstring(self.call.output, name_ctx, **kwargs),
                                                r'%OUTPUT_TYPE%': c_typestring(self.output, name_ctx, **kwargs),
                                                r'%NAME%': integrator_fn_name,
                                                r'%DECORATOR%': '__device__' if is_code_device else '',
                                                r'%NUM_SAMPLES%': f'{num_samples}'
                                            },
                                            name=integrator_fn_name)

        integrator_function.inputs = input_symbols
        integrator_function.output = self.output
        integrator_call_code, integrator_func_list = tegpass_c(
                                                     IR_Call(function=integrator_function,
                                                             inputs=integrator_function.inputs,
                                                             output=integrator_function.output),
                                                     name_ctx,
                                                     **kwargs)
        return integrator_call_code, fn_call_methods + integrator_func_list

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


@overloads(IR_IfElse)
class CPass_IfElse:
    def __tegpass_c__(self, name_ctx, **kwargs):
        if not kwargs.get('no_inline', False):
            # Set C codegen properties
            self.if_call.c_inline = True
            self.else_call.c_inline = True

        if_code, if_functions = tegpass_c(self.if_call, name_ctx, **kwargs)
        else_code, else_functions = tegpass_c(self.else_call, name_ctx, **kwargs)

        # Insert assign statements

        # Use paranthesis if code blocks are empty.
        if not if_code:
            if_code = '{ }'
        if not else_code:
            else_code = '{ }'

        return f'if({c_symbolstring(self.condition, name_ctx, **kwargs)})\n'\
               f'{indent(if_code)}\n'\
               f'else\n'\
               f'{indent(else_code)}\n',\
               [*if_functions, *else_functions]

    def __c_gather_symbols__(self):
        yield from c_gather_symbols(self.output)


@overloads(IR_Call)
class CPass_Call:
    def __tegpass_c__(self, name_ctx, **kwargs):
        if not hasattr(self, 'c_inline'):
            self.c_inline = False

        if not self.c_inline:
            # Function call.
            if hasattr(self.function, 'instrs') and not self.function.instrs:
                # Function is empty. Skip function call.
                if self.output is self.function.output:
                    call_code = ''
                    inner_funclist = []
                else:
                    call_code, inner_funclist = tegpass_c(IR_Assign(self.output, self.function.output),
                                                          name_ctx, **kwargs)
            else:
                input_strings = [f'{c_symbolstring(symbol, name_ctx, **kwargs)},' for symbol in self.inputs]
                input_list = ''.join(input_strings)
                if input_list and input_list[-1] == ',':
                    input_list = input_list[:-1]

                inner_funclist = [self.function]
                output_code = c_symbolstring(self.output, name_ctx, **kwargs)
                call_code = f'{output_code} = {c_symbolstring(self.function, name_ctx, **kwargs)}({input_list});'
        else:
            # Inline function.

            # Add assignment code if necessary
            assign_instrs = []
            for call_input, fn_input in zip(self.inputs, self.function.inputs):
                if call_input is not fn_input:
                    assign_instrs.append(IR_Assign(fn_input, call_input))

            output_assign_instr = ([IR_Assign(self.output, self.function.output)]
                                   if self.output is not self.function.output else [])

            self.function.instrs = assign_instrs + self.function.instrs + output_assign_instr
            self.function.c_inline = True
            self.function.output = self.output

            fn_code, inner_funclist = tegpass_c(self.function, name_ctx, **kwargs)

            input_strings = [f'{c_symbolstring(symbol, name_ctx, **kwargs)},' for symbol in self.inputs]
            input_list = ''.join(input_strings)
            if input_list and input_list[-1] == ',':
                input_list = input_list[:-1]

            inline_lines = ['\t' + line + '\n' for line in fn_code.splitlines(keepends=False)]
            call_code = ''.join(inline_lines)

        return call_code, inner_funclist

    def __c_gather_symbols__(self):
        if self.c_inline or not (hasattr(self.function, 'instrs') and not self.function.instrs):
            yield from c_gather_symbols(self.output)
        elif ((hasattr(self.function, 'instrs') and not self.function.instrs) and
              self.output is not self.function.output):
            yield from c_gather_symbols(self.output)


@overloads(IR_Function)
class CPass_Function:
    def __tegpass_c__(self, name_ctx, **kwargs):
        if not hasattr(self, 'c_inline'):
            self.c_inline = False

        is_code_device = kwargs.get('device_code', False)

        if self.instrs:
            c_instrlist, list_of_c_funclist = zip(*[tegpass_c(instr, name_ctx, **kwargs) for instr in self.instrs])
        else:
            c_instrlist, list_of_c_funclist = '', []

        c_instrlist = [f'\t{c_string}\n' for c_string in c_instrlist if c_string is not None]
        func_body = ''.join(c_instrlist)

        all_symbols = sum([list(c_gather_symbols(instr)) for instr in self.instrs], [])
        """
        if self.label == 'main':
            print('PRINTING')
            for inst in self.instrs:
                print(type(inst))
                print('adding:', list(c_gather_symbols(inst))[0])
        """

        symbols_to_decl = [symbol for symbol in all_symbols if symbol not in self.inputs and symbol is not self.output]

        decl_strings = [f'\t{c_declstring(symbol, name_ctx, **kwargs)};\n'
                        for symbol in symbols_to_decl
                        if c_declstring(symbol, name_ctx, **kwargs)]
        decl_list = ''.join(decl_strings)
        output_decl_string = (f'\t{c_declstring(self.output, name_ctx, **kwargs)};\n' if
                              c_declstring(self.output, name_ctx, **kwargs) else '')

        input_strings = [f'{c_declstring(symbol, name_ctx, **kwargs)},'
                         for symbol in self.inputs
                         if c_declstring(symbol, name_ctx, **kwargs)]
        input_list = ''.join(input_strings)
        if input_list and input_list[-1] == ',':
            input_list = input_list[:-1]

        if not self.c_inline:
            return_string = f'\treturn {c_symbolstring(self.output, name_ctx, **kwargs)};\n'
            output_decl = c_typestring(self.output, name_ctx, **kwargs)

            func_name = c_symbolstring(self, name_ctx, **kwargs)

            decorators = '__device__' if is_code_device else ''

            function = f'{decorators} {output_decl} {func_name}({input_list}){{\n'\
                       f'{decl_list}\n'\
                       f'{output_decl_string}\n'\
                       f'{func_body}\n'\
                       f'{return_string}}}'
        else:
            function = f'{{\n'\
                       f'{decl_list}\n'\
                       f'{func_body}\n'\
                       f'}}'

        c_funclist = sum(list_of_c_funclist, [])
        return function, c_funclist

    def __c_symbolstring__(self, name_ctx, **kwargs):
        if 'func_name' not in dir(self):
            prefix = kwargs.get('fn_name_prefix', '')
            self.func_name = generate_unique_name(name_ctx, label=f'{prefix}{self.label}')

        return self.func_name


def convert_output_to_target(ir_func, target='py'):
    if isinstance(ir_func.output, IR_Variable) and ir_func.output.ctype_mode() == target:
        return

    new_output = IR_Variable(label=ir_func.output.label if hasattr(ir_func.output, 'label') else None,
                             default=ir_func.output.default if hasattr(ir_func.output, 'default') else None,
                             var=ir_func.output._teg_var if hasattr(ir_func.output, 'default') else None)
    new_output.set_ctype_mode(target)

    ir_func.instrs.append(IR_Assign(output=new_output, input=ir_func.output))
    ir_func.output = new_output


def to_c(ir_func, infer_types=True, **kwargs):
    if infer_types:
        infer_typing(ir_func)

    name_ctx = kwargs.pop('name_ctx', {})
    func_code, funcs = tegpass_c(ir_func, name_ctx=name_ctx, **kwargs)

    all_funcs = []
    for func in funcs:
        inner_funcs, arglist = to_c(func, infer_types=False, name_ctx=name_ctx, **kwargs)
        all_funcs.append(inner_funcs)

    all_funcs = sum(all_funcs, [])

    return (all_funcs + [(c_symbolstring(ir_func, name_ctx, **kwargs), func_code)],
            [(inp.label, inp.default, inp._teg_var if hasattr(inp, '_teg_var') else None)
             for inp in ir_func.inputs])
