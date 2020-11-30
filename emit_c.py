from functools import reduce
import operator

from smooth import (
    Sqrt, Sqr
)

FUNC_MAPPING = {
    Sqrt:  (lambda x : f'sqrt({x})'),
    Sqr :  (lambda x : f'pow({x}, 2)')
}

def project_sizes(size1 : int, size2 : int):
    if size1 == 1:
        return size2
    elif size2 == 1:
        return size1
    else:
        assert size1 == size2, f'Size mismatch'
        return size1

class C_SSAVar:
    def __init__(self, name, ctype = None, size = None, assigned = False, default = None):
        self.name = name
        self.ctype = ctype
        self.size = size
        self.assigned = assigned
        self.default = default

    def set_type(self, ctype, size):
        if self.ctype is not None:
            assert self.ctype == ctype, f'"{self.name}" has conflicting types.'
        if self.size is not None:
            assert self.size == size, f'"{self.name}" has conflicting sizes.'

        self.ctype = ctype
        self.size = size

    def require_decl(self):
        self.assigned = True

    def __eq__(self, o):
        return (self.name == o.name) and (self.ctype == o.ctype) and (self.size == o.size)

    def __str__(self):
        size_string = (f"[{self.size}]" if self.size > 1 else "") if self.size is not None else "[X]"
        default_string = f"= {self.default}" if self.default is not None else ""
        assigned_string = 'in' if not self.assigned else ""
        type_string = 'auto' if self.ctype is None else self.ctype
        string = f"{assigned_string} {type_string} {self.name}{size_string} {default_string}"
        return string

class VarContext:
    def __init__(self, _vars = None):
        if _vars is not None:
            self.vars = _vars
        else:
            self.vars = {}

    def get_or_create_var(self, varname, ctype, size):
        if varname not in self.vars.keys():
            self.vars[varname] = C_SSAVar(varname, ctype, size=1)
        return self.vars[varname]

    def get_var(self, varname):
        if varname in self.vars.keys():
            return self.vars[varname]
        return None

    def put_var(self, out_var):
        if out_var.name not in self.vars.keys():
            self.vars[out_var.name] = out_var
        else:
            ov = self.vars[out_var.name]
            assert ov == out_var, \
                f'Variable {out_var.name} has two incompatible types: {out_var.ctype}[{out_var.size}] and {ov.ctype}[{ov.size}]'

    def all_used_vars(self):
        # TODO: Compress
        used_vars = []
        for var in self.vars.values():
            if var.assigned:
                used_vars.append(var)

        return used_vars

    def all_unused_vars(self):
        # TODO: Compress
        unused_vars = []
        for var in self.vars.values():
            if not var.assigned:
                unused_vars.append(var)

        return unused_vars

    def __add__(self, octx):
        new_vars = {}

        for k in octx.vars.keys():
            new_vars[k] = octx.vars[k]

        for k in self.vars.keys():
            if k not in new_vars.keys():
                new_vars[k] = self.vars[k]
            else:
                ov = new_vars[k]
                new_vars[k].assigned = ov.assigned or self.vars[k].assigned

        res = VarContext(new_vars)

        return res

    def __str__(self):
        pstr = "VarContext{\n" 
        for k in self.vars:
            pstr += str(self.vars[k]) + "\n"
        pstr += "}"
        return pstr

class C_SSACode:
    def __init__(self, out_var: C_SSAVar, code = None, ctx: VarContext = None, no_assign: bool = False):
        self.out_var = out_var
        self.code = code

        #if code is not None:
        #    self.out_var.assigned = not no_assign

        if ctx is None:
            ctx = VarContext()

        self.ctx = ctx
        self.ctx.put_var(self.out_var)

        for k in self.ctx.vars.keys():
            assert type(k) is str, f'Inconsistency detected'

    def __add__(self, o):
        if self.code is None:
            return C_SSACode(o.out_var, o.code, o.ctx + self.ctx)
        elif o.code is None:
            return C_SSACode(self.out_var, self.code, o.ctx + self.ctx)
        else:
            return C_SSACode(o.out_var, self.code + '\n' + o.code, o.ctx + self.ctx)

    def __str__(self):
        return f"C_SSACode [{self.out_var}]{{ {self.code} }} \
        (Unassigned: {[var.name for var in self.ctx.all_unused_vars()]}, \
        Assigned: {[var.name for var in self.ctx.all_used_vars()]})"

class C_Method:
    def __init__(self, name = '', code = None, arglist = None, ctype = None, size = None):
        self.name = name
        self.code = code
        self.arglist = arglist if arglist is not None else []
        self.ctype = ctype
        self.size = size

    def __str__(self):

        argstring = ""
        for var in self.arglist:
            argstring = argstring + f'{var},'

        if len(argstring) > 0:
            argstring = argstring[:-1]

        return f"C_Method {self.name}:[{argstring}]->{self.ctype}[{self.size}] {{ {self.code} }}"

class CEmitter:
    def __init__(self, device_code = False, float_type = 'float'):
        self.vars = {}
        self.var_counters = {}
        
        self.device_code = device_code
        self.float_type = float_type

    def _draw_varname(self, prefix="_t_"):
        if prefix not in self.var_counters.keys():
            self.var_counters[prefix] = 0

        num = self.var_counters[prefix]
        self.var_counters[prefix] += 1

        new_varname = f"{prefix}{num}"

        return new_varname

    def literal(self, value = None, out_var = None):
        assert value is not None, f"Literals must have a value"

        try:
            value = float(value)
        except:
            raise ValueError(f'{value} cannot be converted into a real number.')

        ctype = self.float_type
        size = 1

        assert ctype is not None, f"Default values other than float or integer are not supported"

        if out_var is None:
            out_var = self._draw_varname(prefix='_l_')

        out_ssavar = C_SSAVar(out_var, ctype = ctype, size = size, assigned = True)

        return C_SSACode(out_ssavar, f'{out_ssavar.name} = {value};')

    def variable(self, varname: str = None, ctype: str = None, size: int = None, assigned: bool = False, default = None):
        if varname is not None:
            # Make named variable with specific type.
            assert (ctype is not None) and (size is not None), f'For named variables, a type must be provided.'
            out_ssavar = C_SSAVar( varname, 
                                   ctype = ctype, 
                                   size = size, 
                                   assigned = assigned, 
                                   default = default)
            return C_SSACode(out_ssavar, None)
        else:
            # Make variable with random name for intermediate assignment.
            random_name = self._draw_varname()
            out_ssavar = C_SSAVar( random_name, 
                                   assigned = assigned, 
                                   ctype = ctype, 
                                   size = size,
                                   default = default)
            return C_SSACode(out_ssavar, None)


    # Aggregation primitive. Provides additional options for languages where fast 
    # aggregation mechanisms exist. For C, it's the same as using a loop
    def aggregate(self, index_var, lower_code, upper_code, num_code, bind_var, loop_code):
        agg_code = lower_code + upper_code + num_code
        
        # Type inference for dependent variables.
        index_var.set_type('int', 1)
        index_var.require_decl()

        assert bind_var.size == 1, f'Bound aggregation variable must be a scalar'

        step_size = self.variable(assigned = True, ctype=self.float_type, size=1)
        agg_var = self.variable(assigned = True, default = 0).out_var

        step_size.out_var.require_decl()
        bind_var.require_decl()

        # Manually add in additional instructions to aggregate information.
        loop_block = self.block(
                    C_SSACode(step_size.out_var, 
                            f'{step_size.out_var.name} = ({upper_code.out_var.name} - {lower_code.out_var.name}) / ({num_code.out_var.name});') +
                    C_SSACode(bind_var, 
                            f'{bind_var.name} = {lower_code.out_var.name} + {step_size.out_var.name} * ({index_var.name} + 0.5f);') +
                    self.assign(
                        agg_var, 
                        self.mul(
                            self.variable(
                                step_size.out_var.name, 
                                ctype = step_size.out_var.ctype,
                                size = step_size.out_var.size
                            ), 
                            loop_code),
                        op = '+='
                    ) 
                )

        ctx = loop_block.ctx
        ctx.put_var(index_var)
        return agg_code + C_SSACode(loop_block.out_var,
                    f'for({index_var.name} = 0;' +
                        f'{index_var.name} < {num_code.out_var.name};' +
                        f'{index_var.name}++){loop_block.code}', ctx = ctx)

    def assign(self, out_var, code, op = "="):

        out_ssavar = out_var

        out_ssavar.require_decl()

        # If no type, assign type.
        if out_ssavar.ctype is None:
            out_ssavar.set_type(code.out_var.ctype, code.out_var.size)

        if out_ssavar.size == 1:
            return code + C_SSACode(out_ssavar, f'{out_ssavar.name} {op} {code.out_var.name};')
        else:
            if code.out_var.size == 1:
                return code + C_SSACode(out_ssavar, 
                            f'for (uint32_t _iter_ = 0; _iter_ < {out_ssavar.size}; _iter_++) ' + \
                            f'{out_ssavar.name}[_iter_] {op} {code.out_var.name};')
            else:
                return code + C_SSACode(out_ssavar, 
                            f'for (uint32_t _iter_ = 0; _iter_ < {out_ssavar.size}; _iter_++) ' + \
                            f'{out_ssavar.name}[_iter_] {op} {code.out_var.name}[_iter_];')


    def array_assign(self, out_var, code_list):

        for code in code_list:
            assert code.out_var.size == 1, f'Nested tuples cannot be emitted: {code.out_var.name}'

        for code1, code2 in zip(code_list[1:], code_list[:-1]):
            assert code1.out_var.ctype == code2.out_var.ctype, f'Elements in a tuple assignment must be of same type: {code1.out_var.ctype}, {code2.out_var.ctype}'

        out_ssavar = out_var
        out_ssavar.require_decl()

        out_var.set_type(code1.out_var.ctype, len(code_list))

        return reduce(operator.add, code_list) + C_SSACode(out_ssavar,
                        ''.join([f'{out_ssavar.name}[{idx}] = {code.out_var.name};\n' for idx, code in enumerate(code_list)])
                        )

    def condition(self, cond, if_body, else_body, out_var = None):
        # TODO: Temporary assertion while refactoring code.
        assert out_var is None

        # Infer type.
        output_size = project_sizes(if_body.out_var.size, else_body.out_var.size)
        output_type = if_body.out_var.ctype
        assert if_body.out_var.ctype == else_body.out_var.ctype, f'Condition branches have different output types'

        out_var = self.variable(ctype = output_type, size = output_size).out_var

        # Make block code.
        if_block = self.block(self.assign(out_var, if_body))
        else_block = self.block(self.assign(out_var, else_body))

        return cond + \
            C_SSACode(out_var, 
                f'if ({cond.out_var.name})\n{if_block.code} \
                else \n{else_block.code}',
                ctx = if_block.ctx + else_block.ctx
            )

    def block(self, in_code):
        # Get a list of all free variables that appear on the LHS.
        used_list = in_code.ctx.all_used_vars()

        # Declare all of them at the beginning of the block.
        decl_list = []
        for var in used_list:
            if var.name == in_code.out_var.name:
                # Skip over the output variable of this block.
                continue

            if var.size == 1:
                if var.default is None:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name};'))
                else:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name} = {var.default};'))
            else:
                if var.default is None:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name}[{var.size}];'))
                else:
                    init_string = ''.join([f'{var.default},' for i in range(var.size)])[:-1]
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name}[{var.size}] = {{{init_string}}};'))

        if len(decl_list) > 0:
            all_decls = reduce(operator.add, decl_list)
        else:
            all_decls = None

        unused_list = in_code.ctx.all_unused_vars()
        unused_dict = {}
        for var in unused_list:
            unused_dict[var.name] = var

        unused_ctx = VarContext(unused_dict)

        decl_code = f'{all_decls.code}' if all_decls is not None else ''
        # Build the block code with all the declarations.
        block_code = C_SSACode(in_code.out_var, 
                                f'{{{decl_code}\n{in_code.code}\n}}', 
                                ctx = unused_ctx)
        return block_code
    
    def method(self, in_code, method_name = 'method', arglist = []):
        # Get a list of all free variables that appear on the LHS.
        used_list = in_code.ctx.all_used_vars()
        
        # Declare all of them at the beginning of the block.
        decl_list = []
        for var in used_list:
            if var.size == 1:
                if var.default is None:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name};'))
                else:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name} = {var.default};'))
            else:
                if var.default is None:
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name}[{var.size}];'))
                else:
                    init_string = ''.join([f'{var.default},' for i in range(var.size)])[:-1]
                    decl_list.append(C_SSACode(var, f'{var.ctype} {var.name}[{var.size}] = {{{init_string}}};'))

        all_decls = reduce(operator.add, decl_list)

        args = { var.name : var for var in in_code.ctx.all_unused_vars() }

        # print(args.keys())

        for name in arglist:
            assert name in args.keys(), f'Couldn\'t find argument symbol "{name}" in the computation tree'

        full_arglist = [ args[name] for name in arglist ]
        full_arglist = full_arglist + [ args[name] for name in args.keys() if name not in arglist ]

        # Build function argument string.
        # Don't use __str__ for variables.
        argstring = ""
        for var in full_arglist:
            if var.size > 1:
                argstring = argstring + f'{var.ctype} {var.name}[{var.size}],'
            else:
                if var.default is None:
                    argstring = argstring + f'{var.ctype} {var.name},'
                else:
                    argstring = argstring + f'{var.ctype} {var.name} = {var.default},'

        # Remove trailing comma.
        if len(argstring) > 1:
            argstring = argstring[:-1]

        size_str = f"[{in_code.out_var.size}]" if in_code.out_var.size > 1 else ""

        result_type = f'{method_name}_result'

        decl_type_string = '__device__' if self.device_code else ''
        if in_code.out_var.size > 1:
            result_struct = f'{decl_type_string} struct {result_type}{{ {in_code.out_var.ctype} o[{in_code.out_var.size}]; }};'
            inner_assign_statement = "".join([f"{in_code.out_var.name}[{i}]," for i in range(in_code.out_var.size)])[:-1]
            return_statement = f'return {result_type}{{ {{ {inner_assign_statement} }} }};'
        else:
            result_struct = f'typedef {in_code.out_var.ctype} {result_type};'
            return_statement = f'return {in_code.out_var.name};'

        method_decorator_string = '__device__' if self.device_code else ''
        # Build the method code with all the declarations.
        method_code = C_Method(method_name,
                                f'{result_struct}\n' +
                                f'{method_decorator_string} ' +
                                f'{result_type} ' +
                                f'{method_name} ({argstring}){{' + 
                                f'\n{all_decls.code}\n{in_code.code}\n' +
                                f'{return_statement}}}', 
                                arglist,
                                in_code.out_var.ctype,
                                in_code.out_var.size
                            )
        return method_code

    # Pretty sure this shouldn't refer to 'smooth'.
    # Handle all of this when rewriting these passes.
    def smooth_func(self, code1, out_var = None, op = None):
        output_size = code1.out_var.size
        output_type = code1.out_var.ctype

        output = self.variable(ctype = output_type, size = output_size).out_var
        output.require_decl()

        instr_generator = FUNC_MAPPING[op]

        if output.size == 1:
            return code1 + C_SSACode(output, 
                f'{output.name}  = ' + instr_generator(f'{code1.out_var.name}') + ';'
            )
        else:
            return code1 + C_SSACode(output, 
                f'for (uint32_t _iter_ = 0; _iter_ < {output.size}; _iter_++) ' + \
                f'{output.name}[_iter_] = ' + instr_generator(f'{code1.out_var.name}[_iter_]') + ';'
            )

    def _binary_op_(self, code1, code2, out_var = None, op = '+', out_type = None):
        # TODO: Temporary assertion for code refactor
        assert out_var is None

        # Infer type
        output_size = project_sizes(code1.out_var.size, code2.out_var.size)
        output_type = code1.out_var.ctype if out_type is None else out_type
        assert code1.out_var.ctype == code2.out_var.ctype, f'Operands are of incompatible types'

        output = self.variable(ctype = output_type, size = output_size).out_var
        output.require_decl()

        if output.size == 1:
            return code1 + code2 + C_SSACode(output, 
                f'{output.name} = {code1.out_var.name} {op} {code2.out_var.name};'
            )
        else:
            if code1.out_var.size == 1:
                code1_access = f'{code1.out_var.name}'
            else:
                code1_access = f'{code1.out_var.name}[_iter_]'

            if code2.out_var.size == 1:
                code2_access = f'{code2.out_var.name}'
            else:
                code2_access = f'{code2.out_var.name}[_iter_]'
            
            return code1 + code2 + C_SSACode(output, 
                f'for (uint32_t _iter_ = 0; _iter_ < {output.size}; _iter_++) ' + \
                f'{output.name}[_iter_] = {code1_access} {op} {code2_access};'
            )

    def mul(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '*')

    def divide(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '/')

    def add(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '+')

    def logical_and(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '&&')

    def logical_or(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '||')

    def invert(self, code1, out_var = None):
        return self._binary_op_(self.literal(1.0), code1, out_var, op = '/')

    def subtract(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '-')

    def compare_lt(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '<', out_type = 'bool')

    def compare_gt(self, code1, code2, out_var = None):
        return self._binary_op_(code1, code2, out_var, op = '>', out_type = 'bool')
