import os
import subprocess
import importlib

from teg.passes.compile import to_ir
from teg.ir.passes.to_c import to_c, convert_output_to_target
from teg.ir.passes.to_c import CTemplateFunction

from teg import ITeg
from teg import __data_path__, __include_path__

from .eval_mode import EvalMode


def _get_command_result(command):
    proc = os.popen(command)
    output = proc.read()
    proc.close()
    return output


class C_EvalMode(EvalMode):

    _module_count_ = 0

    def __init__(self, expr: ITeg, num_samples=50):
        super(C_EvalMode, self).__init__(name='C')

        self.ir_func = to_ir(expr)
        convert_output_to_target(self.ir_func, 'native')  # Ensure output type is generic_array<float, N>

        self.options = {'num_samples': num_samples, 'float_width': 'float'}
        funclist, arglist = to_c(self.ir_func, **self.options)
        self.funclist = funclist
        self.arglist = arglist

        assert len(self.funclist) > 0, 'Code generation failed. Cannot find any routines'

        self.outfolder = '/tmp/teg_cpp'
        if not os.path.exists(self.outfolder):
            os.mkdir(self.outfolder)

        self.main_filename = self.outfolder + f'/main_{C_EvalMode._module_count_}.cc'
        self.out_filename = self.outfolder + f'/tegout_{C_EvalMode._module_count_}.h'
        self.module_name = f'teg_jit_module_{C_EvalMode._module_count_}'
        self.module_filename = None

        # Status variables
        self.preprocessed = False
        self.compiled = False
        self.loaded = False

        self.module = None

        # Note: NOT thread-safe
        C_EvalMode._module_count_ += 1

    def _make_main_file(self):
        fn_name, fn_body = self.funclist[-1]

        headers = f'#include <iostream>\n'\
                  f'#include <string>\n'\
                  f'#include <math.h>\n'\
                  f'#include "{self.out_filename}"\n'\

        float_width = self.options['float_width']

        string_to_float = 'stof' if float_width == 'float' else 'stod'
        input_string = ''.join([f'\t{float_width} {arg[0]} = {string_to_float}(std::string(argv[{idx + 1}]));\n'
                                for idx, arg in enumerate(self.arglist)])

        output_var = '__output__'
        output_size = self.ir_func.output.irtype().size
        output_type = None
        if output_size == 1:
            output_string = f'\tstd::cout << {output_var} << std::endl;\n'
            output_type = f'{float_width}'
        else:
            output_string = f'\tfor(int __iter__ = 0; __iter__ < {output_size}; __iter__++)'\
                            f'\t\tstd::cout << {output_var}[__iter__] << std::endl;\n'
            output_type = f'generic_array<{float_width}, {output_size}>'

        call_list = ','.join([f'{arg[0]}' for arg in self.arglist])
        call_string = f'{output_type} {output_var} = {fn_name}({call_list});'

        main_code = f'{headers}'\
                    f'int main(int argc, char** argv){{'\
                    f'\tif(argc != {len(self.arglist) + 1})'\
                    f'\t\tstd::cout << "Invalid number of inputs." << std::endl;'\
                    f'{input_string}\n'\
                    f'{call_string}\n'\
                    f'{output_string}\n'\
                    f'return 0;\n'\
                    f'}}'

        _outfile = open(self.main_filename, 'w')
        _outfile.write(main_code)
        _outfile.close()

    def _preprocess(self):
        code = ''
        for func_name, func_body in self.funclist:
            code += func_body + '\n'

        header = '#include "teg_c_runtime.h"'
        _outfile = open(self.out_filename, "w")
        _outfile.write(f'{header}\n{code}')
        _outfile.close()

        self._make_main_file()

        self.preprocessed = True

    def _compile(self):
        teg_runtime_includes = f'-I{__include_path__}'

        self.module_filename = f'{self.outfolder}/{self.module_name}'

        compile_command = f"g++ -O3 -std=c++11 -fPIC {teg_runtime_includes} "\
                          f"{self.main_filename} -o {self.module_filename}"
        proc = subprocess.Popen(compile_command,
                                stdout=subprocess.PIPE, shell=True)

        (out, err) = proc.communicate()
        assert err is None, f'Error: {err}\nOutput: {out}'

        self.compiled = True

    def eval(self, bindings={}, **kwargs):
        if not self.preprocessed:
            self._preprocess()
        if not self.compiled:
            self._compile()

        bindings = {f'{k.name}_{k.uid}': v for k, v in bindings.items()}
        args = [bindings[arg] if arg in bindings.keys() else teg_var.value for arg, default, teg_var in self.arglist]

        assert os.path.exists(self.module_filename), f'Could not find binary {self.module_filename}'
        assert all([arg is not None for arg in args]),\
               f'No bindings found for {[self.arglist[i][2] for i, arg in enumerate(args) if arg is None]}'

        run_command = f"{self.module_filename} {' '.join([format(arg) for arg in args])}"
        proc = subprocess.Popen(run_command,
                                stdout=subprocess.PIPE, shell=True)

        (out, err) = proc.communicate()
        assert err is None, f'Error: {err}\nOutput: {out}'

        out_size = self.ir_func.output.irtype().size
        if out_size > 1:
            lines = out.splitlines(keepends=False)
            assert len(lines) == out_size,\
                   f'Program returned unexpected number of outputs {len(lines)}, expected: {out_size}'
            floatlines = [float(line) for line in lines]
            return floatlines
        else:
            return float(out.rstrip())


class C_EvalMode_PyBind(EvalMode):

    _module_count_ = 0

    def __init__(self, expr: ITeg, num_samples=50):
        super(C_EvalMode_PyBind, self).__init__(name='C')

        ir_func = to_ir(expr)

        convert_output_to_target(ir_func, 'py')

        options = {'num_samples': num_samples}
        funclist, arglist = to_c(ir_func, **options)
        self.funclist = funclist
        self.arglist = arglist

        assert len(self.funclist) > 0, 'Code generation failed. Cannot find any routines'

        self.outfolder = '/tmp/teg_cpp'
        if not os.path.exists(self.outfolder):
            os.mkdir(self.outfolder)

        self.pybind_filename = self.outfolder + '/bind.c'
        self.out_filename = self.outfolder + '/tegout.h'
        self.module_name = f'teg_jit_module_{C_EvalMode_PyBind._module_count_}'
        self.module_filename = None

        # Status variables
        self.preprocessed = False
        self.compiled = False
        self.loaded = False

        self.module = None

        # Note: NOT thread-safe
        C_EvalMode_PyBind._module_count_ += 1

    def _make_pybind_file(self):
        fn_name, fn_body = self.funclist[-1]

        _template_file = open(__data_path__ + '/templates/pybind.template.c', 'r')
        template = _template_file.read()
        _template_file.close()
        pybind_template = CTemplateFunction(template, {
                                        r'%C_FILENAME%': self.out_filename,
                                        r'%FN_PYTHON_NAME%': 'eval',
                                        r'%FN_C_NAME%': fn_name,
                                        r'%MODULE_NAME%': self.module_name
                                        })

        pybind_file_body, _ = pybind_template.__tegpass_c__(name_ctx=None)
        _outfile = open(self.pybind_filename, 'w')
        _outfile.write(pybind_file_body)
        _outfile.close()

    def _preprocess(self):
        code = ''
        for func_name, func_body in self.funclist:
            code += func_body + '\n'

        header = '#include "teg_c_runtime.h"'
        _outfile = open(self.out_filename, "w")
        _outfile.write(f'{header}\n{code}')
        _outfile.close()

        self._make_pybind_file()

        self.preprocessed = True

    def _compile(self):
        pybind_includes = _get_command_result('python3 -m pybind11 --includes').rstrip()
        teg_runtime_includes = f'-I{__include_path__}'
        extension_suffix = _get_command_result('python3-config --extension-suffix').rstrip()

        self.module_filename = f'{self.outfolder}/{self.module_name}{extension_suffix}'

        compile_command = f"g++ -O3 -shared -std=c++11 -fPIC {pybind_includes} {teg_runtime_includes} "\
                          f"{self.pybind_filename} -o {self.module_filename}"
        proc = subprocess.Popen(compile_command,
                                stdout=subprocess.PIPE, shell=True)

        (out, err) = proc.communicate()
        assert err is None, f'Error: {err}\nOutput: {out}'

        self.compiled = True

    def _load(self):
        spec = importlib.util.spec_from_file_location(self.module_name, self.module_filename)
        assert spec is not None, print(f'Could not load python JIT module: {self.module_filename}')
        code_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(code_module)

        self.module = code_module

        # Clean up to prevent bloat.
        os.remove(self.module_filename)
        os.remove(self.pybind_filename)
        # os.remove(self.out_filename)
        self.loaded = True

    def eval(self, bindings={}, **kwargs):
        if not self.preprocessed:
            self._preprocess()
        if not self.compiled:
            self._compile()
        if not self.loaded:
            self._load()

        bindings = {f'{k.name}_{k.uid}': v for k, v in bindings.items()}
        args = [bindings[arg] if arg in bindings.keys() else teg_var.value for arg, default, teg_var in self.arglist]

        assert all([arg is not None for arg in args]),\
               f'Could not resolve all arguments {zip(args, kwargs.keys())}'

        output = self.module.eval(*args)
        assert output is not None, f'Eval failed with inputs: {args}'
        return output
