import subprocess
import os
from teg import __temp_path__

def runProgram(program, silent=False):
    prog_name, out_size = program
    proc = subprocess.Popen([prog_name], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err is not None:
        print(f'Error: {err}')
        print(f'Output: {out}')

    if out_size > 1:
        return [float(line) for line in out.decode().split('\n')[:out_size]]
    else:
        return float(out)


def compileProgram(program, silent=True):
    fn_name, arglist, code, out_size = (program.name, program.arglist, program.code, program.size)

    for arg in arglist:
        assert arg.default is not None, f'Var {arg.name} has no default value set, which is currently not supported. '

    # Build dummy main function
    if out_size == 1:
        main_code = (f' \n'
                     f'int main(int argc, char** argv){{\n'
                     f'  std::cout << {fn_name}();\n'
                     f'  return 0;\n'
                     f'}}')
    else:
        main_code = (f' \n'
                     f'int main(int argc, char** argv){{\n'
                     f'  {fn_name}_result s = {fn_name}();\n'
                     f'  for(int i = 0; i < {out_size}; i++){{'
                     f'      std::cout << s.o[i] << std::endl;\n'
                     f'  }}'
                     f'  return 0;\n'
                     f'}}')

    # Include basic IO
    header_code = '#include <iostream>\n' + '#include <math.h>\n'

    all_code = header_code + code + main_code
    cppfile = open(f'{__temp_path__}{os.path.sep}_teg_cpp_out.cpp', 'w')
    cppfile.write(all_code)
    cppfile.close()

    filename = f'{__temp_path__}{os.path.sep}_teg_cpp_out'
    proc = subprocess.Popen(f'g++ -std=c++11 {filename}.cpp -o {filename} -O3', stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()

    if err is not None:
        print(f'Error: {err}')
        print(f'Output: {out}')

    return filename, out_size
