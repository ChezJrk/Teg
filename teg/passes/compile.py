from typing import List, Dict

from teg import (
    ITeg,
    Const,
    Var,
    Add,
    Mul,
    IfElse,
    Teg,
    SmoothFunc,
    Tup,
    LetIn,
    Bool,
    And,
    Or,
    Invert,
)
from teg.derivs import FwdDeriv, RevDeriv

from teg.ir.instr import (
    IR_Instruction,
    IR_Symbol,
    IR_Literal,
    IR_Variable,
    IR_Function,
    IR_Call,

    IR_Add,
    IR_Mul,
    IR_Divide,
    IR_UnaryMath,
    IR_IfElse,
    IR_Integrate,
    IR_Pack,
    IR_CompareLT,
    IR_LAnd,
    IR_LOr
)


def to_ir(expr: ITeg) -> IR_Function:
    instr_list, out_symbol, free_symbols = _to_ir(expr, {})
    return IR_Function(instr_list, output=out_symbol, inputs=free_symbols.values(), label='main')


def _to_ir(expr: ITeg, symbols: Dict[str, IR_Symbol]) -> (List[IR_Instruction], IR_Variable, Dict[str, IR_Symbol]):

    if isinstance(expr, Const):
        return [], IR_Literal(value=expr.value), {}

    elif isinstance(expr, Var):
        label = f'{expr.name}_{expr.uid}'
        symbol = symbols.get(label, IR_Variable(label=label, var=expr, default=expr.value))
        return ([],
                symbol,
                {label: symbol})

    elif isinstance(expr, Invert):
        child_code, code_var, free_symbols = _to_ir(expr.child, symbols)
        out_var = IR_Variable()

        code = [*child_code, IR_Divide(out_var, IR_Literal(1.0), code_var)]
        return code, out_var, free_symbols

    elif isinstance(expr, SmoothFunc):
        expr_code, expr_var, free_symbols = _to_ir(expr.expr, symbols)
        out_var = IR_Variable()

        code = [*expr_code, IR_UnaryMath(out_var, expr_var, fn_class=expr.__class__)]
        return code, out_var, free_symbols

    elif isinstance(expr, IfElse):
        cond_code, cond_var, cond_symbols = _to_ir(expr.cond, symbols)
        if_code, if_var, if_symbols = _to_ir(expr.if_body, {**symbols, **cond_symbols})
        else_code, else_var, else_symbols = _to_ir(expr.else_body, {**symbols, **cond_symbols, **if_symbols})

        if_fn = IR_Function(if_code, output=if_var, inputs=if_symbols.values(), label='if_block')
        else_fn = IR_Function(else_code, output=else_var, inputs=else_symbols.values(), label='else_block')

        out_var = IR_Variable()
        return ([*cond_code,
                 IR_IfElse(output=out_var,
                           condition=cond_var,
                           if_call=IR_Call(output=out_var, inputs=if_symbols.values(), function=if_fn),
                           else_call=IR_Call(output=out_var, inputs=else_symbols.values(), function=else_fn))],
                out_var,
                {**cond_symbols, **if_symbols, **else_symbols})

    elif isinstance(expr, Teg):
        lower_code, lower_var, lower_symbols = _to_ir(expr.lower, symbols)
        upper_code, upper_var, upper_symbols = _to_ir(expr.upper, {**symbols, **lower_symbols})

        body_code, body_var, body_symbols = _to_ir(expr.body, {**symbols, **lower_symbols, **upper_symbols})

        teg_var = expr.dvar
        teg_var_label = f"{teg_var.name}_{teg_var.uid}"

        if teg_var_label not in body_symbols.keys():
            body_symbols = {**body_symbols,
                            teg_var_label: IR_Variable(label=teg_var_label, var=teg_var, default=None)}

        body_fn = IR_Function(body_code, output=body_var, inputs=body_symbols.values(), label='body_block')

        out_var = IR_Variable()

        return ([*lower_code,
                 *upper_code,
                 IR_Integrate(output=out_var,
                              call=IR_Call(output=body_var, inputs=body_symbols.values(), function=body_fn),
                              teg_var=body_symbols[teg_var_label],
                              lower_bound=lower_var,
                              upper_bound=upper_var)],
                out_var,
                {**upper_symbols, **lower_symbols,
                 **{label: symbol for label, symbol in body_symbols.items() if label != teg_var_label}})

    elif isinstance(expr, Tup):
        all_free_symbols = {}
        all_instrs = []
        child_vars = []

        for child in expr.children:
            child_list, child_var, child_free_symbols = _to_ir(child, {**symbols, **all_free_symbols})

            all_free_symbols = {**all_free_symbols, **child_free_symbols}
            all_instrs.extend(child_list)
            child_vars.append(child_var)

        # child_lists, child_vars, child_symbols = zip(*[ for child in expr.children]) # Incorrect
        out_var = IR_Variable()
        # print([str(child) for child in child_vars])
        return (all_instrs + [IR_Pack(output=out_var, inputs=child_vars)],
                out_var,
                all_free_symbols)

    elif isinstance(expr, LetIn):
        expr_free_symbols = {}
        all_instrs = []
        expr_vars = []
        for child in expr.new_exprs:
            expr_list, expr_var, free_symbols = _to_ir(child, {**symbols, **expr_free_symbols})
            expr_free_symbols = {**expr_free_symbols, **free_symbols}
            all_instrs.extend(expr_list)
            expr_vars.append(expr_var)

        new_symbols = {f'{var.name}_{var.uid}': symbol for var, symbol in zip(expr.new_vars, expr_vars)
                       if not isinstance(symbol, IR_Literal)}
        ctx_symbols = {**symbols, **expr_free_symbols, **new_symbols}

        body_list, body_var, body_symbols = _to_ir(expr.expr, ctx_symbols)

        body_fn = IR_Function(body_list, output=body_var, inputs=body_symbols.values(), label='body_block')
        body_call = IR_Call(output=body_var, inputs=body_symbols.values(), function=body_fn)

        return ([*all_instrs, body_call],
                body_var,
                {**expr_free_symbols,
                 **{label: symbol for label, symbol in body_symbols.items() if label not in new_symbols.keys()}})

    elif isinstance(expr, (FwdDeriv, RevDeriv)):
        return _to_ir(expr.deriv_expr, symbols)

    elif isinstance(expr, (Add, Mul)):
        assert len(expr.children) == 2, f'Binary expression "{expr.name}" has an invalid number of operands'

        ir_left, left_var, left_symbols = _to_ir(expr.children[0], symbols)
        ir_right, right_var, right_symbols = _to_ir(expr.children[1], {**symbols, **left_symbols})

        out_var = IR_Variable()

        if isinstance(expr, Add):
            code = [*ir_left, *ir_right, IR_Add(out_var, left_var, right_var)]
        elif isinstance(expr, Mul):
            code = [*ir_left, *ir_right, IR_Mul(out_var, left_var, right_var)]

        return code, out_var, {**left_symbols, **right_symbols}

    elif isinstance(expr, (Bool, And, Or)):
        ir_left, left_var, left_symbols = _to_ir(expr.left_expr, symbols)
        ir_right, right_var, right_symbols = _to_ir(expr.right_expr, {**symbols, **left_symbols})

        out_var = IR_Variable()

        if isinstance(expr, And):
            code = [*ir_left, *ir_right, IR_LAnd(out_var, left_var, right_var)]
        elif isinstance(expr, Or):
            code = [*ir_left, *ir_right, IR_LOr(out_var, left_var, right_var)]
        elif isinstance(expr, Bool):
            code = [*ir_left, *ir_right, IR_CompareLT(out_var, left_var, right_var)]

        return code, out_var, {**left_symbols, **right_symbols}

    else:
        raise ValueError(f'The type of the expr "{type(expr)}" does not have a supported derivative.')
