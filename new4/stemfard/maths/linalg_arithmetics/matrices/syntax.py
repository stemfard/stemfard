from numpy import array
from sympy import Matrix

from stemfard.maths.linalg_arithmetics.matrices.utils import (
    MAP_OPERATION,
    MAP_SYMBOL,
    MatrixOperations
)
from stemfard.core.io.convert_data import DataConverter
from stemfard.core.models import Syntaxes


def _header(operation) -> list[str]:
        
    syntax_list: list[str] = []
    op = MAP_OPERATION[operation]
    symbol = MAP_SYMBOL[operation]
    if len(MAP_OPERATION) > 1:
        syntax_list.append(f"# {op}"),
        syntax_list.append(f'print("\\n{op}\\n")')
        
    return syntax_list, symbol


def _display(n: int) -> str:
    if n > 1:
        return "display(result)\n"
    return "result\n"


def _sympy(
    A: Matrix,
    B: Matrix,
    operations: MatrixOperations,
    param_names: tuple[str, str]
) -> str:

    syntax_list: list[str] = []
    converter = DataConverter()
    param_a, param_b = param_names
    n = len(operations)
    new_line = "\n" if n > 1 else ""
    
    syntax_list.append("# Create the two matrices")
    syntax_list.append(f"{param_a} = sym.Matrix({converter.to_python(A)})")
    syntax_list.append(
        f"{param_b} = sym.Matrix({converter.to_python(B)}){new_line}"
    )
    
    if "add" in operations:
        header, symbol = _header("add")
        syntax_list.extend(header) 
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "subtract" in operations:
        header, symbol = _header("subtract")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))
        

    if "multiply" in operations:
        header, symbol = _header("multiply")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "divide" in operations:
        header, symbol = _header("divide")
        syntax_list.extend(header)
        syntax_list.append(f"nrows, ncols = {param_a}.shape")
        syntax_list.append(
            f"result = sym.Matrix([[{param_a}[i , j] / {param_b}[i , j] "
            f"for j in range(ncols)] for i in range(nrows)])"
        )
        syntax_list.append(_display(n=n))

    if "raise" in operations:
        header, symbol = _header("raise")
        syntax_list.extend(header)
        syntax_list.append(f"nrows, ncols = {param_a}.shape")
        syntax_list.append(
            f"result = sym.Matrix([[{param_a}[i , j] ** {param_b}[i , j] "
            f"for j in range(ncols)] for i in range(nrows)])"
        )
        syntax_list.append(_display(n=n))

    if "matmul" in operations:
        header, symbol = _header("matmul")
        syntax_list.extend(header)  
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))
        
    return "\n".join(syntax_list)


def _numpy(
    A: Matrix,
    B: Matrix,
    operations: MatrixOperations,
    param_names: tuple[str, str]
) -> str:

    syntax_list: list[str] = []
    converter = DataConverter()
    param_a, param_b = param_names
    n = len(operations)
    new_line = "\n" if n > 1 else ""
    
    try:
        # will crush if non-numeric
        A = array(Matrix(A), dtype=float).tolist()
        B = array(Matrix(B), dtype=float).tolist()
    except (TypeError, ValueError):
        return None
    
    syntax_list.append("# Create the two matrices")
    syntax_list.append(f"{param_a} = np.array({converter.to_python(A)})")
    syntax_list.append(
        f"{param_b} = np.array({converter.to_python(B)}){new_line}"
    )
    
    if "add" in operations:
        header, symbol = _header("add")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "subtract" in operations:
        header, symbol = _header("subtract")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "multiply" in operations:
        header, symbol = _header("multiply")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "divide" in operations:
        header, symbol = _header("divide")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "raise" in operations:
        header, symbol = _header("raise")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))

    if "matmul" in operations:
        header, symbol = _header("matmul")
        syntax_list.extend(header)
        syntax_list.append(f"result = {param_a} {symbol} {param_b}")
        syntax_list.append(_display(n=n))
        
    return "\n".join(syntax_list)


def generate_syntax(
    A: Matrix,
    B: Matrix,
    operations: MatrixOperations,
    param_names: tuple[str, str]    
) -> Syntaxes:
    
    params = {
        "A": A,
        "B": B,
        "operations": operations,
        "param_names": param_names
    }
    
    return Syntaxes(
        numpy=_numpy(**params),
        sympy=_sympy(**params)
    )