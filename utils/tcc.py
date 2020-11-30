from optparse import OptionParser
import importlib.util

from teg.ir import emit
from teg.passes import simplify

parser = OptionParser()
parser.add_option("-o", "--output", dest="output", help="output file")
parser.add_option("-t", "--target", dest="target", help="language target", default='C')
parser.add_option("-f", "--fprecision", dest="precision", help="language target", default='single')
parser.add_option("-m", "--method", dest="method", help="method name", default='teg_method')
parser.add_option("-s", "--samples", dest="samples", help="sample count", default=50)

(options, args) = parser.parse_args()

float_type = 'float' if options.precision == 'single' else 'double'

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


if '__ARGLIST__' in dir(code_module):
    arglist = code_module.__ARGLIST__
else:
    arglist = []

print(f"Converting to {options.target} file {output_file}")
method = emit(program,
              target=options.target,
              fn_name=options.method,
              arglist=[f'{var.name}_{var.uid}' for var in arglist],
              num_samples=options.samples,
              float_type=float_type)

ofile = open(output_file, "w")
ofile.write(method.code)
print(f"Generated {options.target} code in {output_file}")
ofile.close()
