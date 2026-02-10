from sympy import Matrix

from stemfard.maths.linalg_arithmetics.matrices.utils import MatrixOperations


def question_str(
    a_latex: str,
    b_latex: str,
    operation: MatrixOperations,
    param_names: tuple[str, str]
) -> str:

    ops: list[str] = []
    
    if "add" in operation:
        ops.append("sum")

    if "subtract" in operation:
        ops.append("difference")

    if "multiply" in operation:
        ops.append("element-wise product")

    if "divide" in operation:
        ops.append("element-wise quotient")

    if "raise" in operation:
        ops.append("element-wise power")

    if "matmul" in operation:
        ops.append("matrix product")

    param_a, param_b = param_names
    
    if len(ops) == 1:
        ops_latex = f"\\( \\text{{{ops[0]}}} \\)"
    else:
        ops_latex = (
            f"\\( \\text{{{', '.join(ops[:-1])}}} \\) and "
            f"\\( \\text{{{ops[-1]}}} \\)"
        )

    qtn = [
        f"Compute the {ops_latex} of the matrices "
        f"\\( {param_a} \\) and \\( {param_b} \\) given below.",
        f"\\[ {param_a} = {a_latex} \\:, \\quad {param_b} = {b_latex} \\]"
    ]

    return qtn