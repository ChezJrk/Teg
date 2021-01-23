from optparse import OptionParser
import sys
import importlib.util

from teg.passes import compile
from teg.passes.simplify import simplify
from teg.ir.passes.to_c import to_c
from teg import __include_path__


parser = OptionParser()
parser.add_option("-I", "--includes", dest="includes",
                  help="Returns C/C++ include directory for Teg",
                  action='store_true',
                  default=False)
parser.add_option("-i", "--include-options", dest="include_options",
                  help="Returns C/C++ include options for Teg",
                  action='store_true',
                  default=False)
parser.add_option("-c", "--compile", dest="compile",
                  help="Compiles a command",
                  action='store_true',
                  default=False)

parser.add_option("-o", "--output", dest="output", help="output file", default=None)
parser.add_option("-t", "--target", dest="target", help="language target", default='C')
parser.add_option("-f", "--fprecision", dest="precision", help="language target", default='single')
parser.add_option("-m", "--method", dest="method", help="method name", default='teg_method')
parser.add_option("-s", "--samples", dest="samples", help="sample count", default=50)


(options, args) = parser.parse_args()

assert [options.includes, options.include_options, options.compile].count(True) == 1,\
       'Exactly one of --includes, --include-options and --compile can be specified'

if options.include_options:
    if ' ' in __include_path__:
        print(f'-I"{__include_path__}"')
    else:
        print(f'-I{__include_path__}')
    sys.exit(0)

elif options.includes:
    print(__include_path__)
    sys.exit(0)

elif options.compile:
    assert options.precision in ['single', 'double'],\
        f'Invalid precision setting {options.precision}. Specify either single or double'
    float_type = {'single': 'float',
                  'double': 'double'}[options.precision]

    input_file = args[0]
    print(input_file)
    output_file = options.output

    spec = importlib.util.spec_from_file_location("module.name", input_file)
    print(f"Building program {input_file}")
    code_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_module)

    assert '__PROGRAM__' in dir(code_module), f'Unable to find "__PROGRAM__" variable in {input_file}'

    program = code_module.__PROGRAM__
    print(f"Simplifying program {input_file}")
    program = simplify(program)

    # print(f"CSE {input_file}")
    # program = cse(program)

    if '__ARGLIST__' in dir(code_module):
        arglist = code_module.__ARGLIST__
    else:
        arglist = []

    print(f"Converting to {options.target} file {output_file}")

    assert options.target in ['C', 'CUDA_C'], 'No other backends are supported at the moment'

    ir_func = compile.to_ir(program)
    # Force name.
    ir_func.func_name = options.method

    # Force input order.
    labelled_inputs = {symbol.label: symbol for symbol in ir_func.inputs}
    ordered_inputs = [labelled_inputs[f'{var.name}_{var.uid}'] for var in arglist]
    ir_func.inputs = ordered_inputs + list(set(ir_func.inputs) - set(ordered_inputs))

    if options.target == 'C':
        funclist, arglist = to_c(ir_func,
                                 float_width=float_type,
                                 num_samples=options.samples,
                                 fn_name_prefix=options.method)
        header = '#include "teg_c_runtime.h"'
        code = header + '\n'
        for func_name, func_body in funclist:
            code += func_body + '\n'

    elif options.target == 'CUDA_C':
        funclist, arglist = to_c(ir_func,
                                 device_code=True,
                                 float_width=float_type,
                                 num_samples=options.samples,
                                 fn_name_prefix=options.method)
        header = '#include "teg_cuda_runtime.h"'
        code = header + '\n'
        for func_name, func_body in funclist:
            code += func_body + '\n'

    ofile = open(output_file, "w")
    ofile.write(code)
    print(f"Generated {options.target} code in {output_file}")
    ofile.close()