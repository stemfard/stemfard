from stemfard.maths.linalg_arithmetics.matrices._base import BaseLinalgMatrixArithmetics
from stemfard.core._strings import str_color_math
from stemfard.core.arrays_general import general_array, general_array_ab, generate_a_times_b


class MatrixArithmeticsBackground(BaseLinalgMatrixArithmetics):
    """Background"""
    
    @property
    def _B(self) -> str:
        return  str_color_math('B')
    
    
    def _introduction(self, operation: str) -> str:
        
        steps_mathjax: list[str] = []
        B = str_color_math('B')
        b_ij = str_color_math(f'b_{{ij}}')
        
        _map = {
            "add": ("adding", "+"),
            "subtract": ("subtracting", "-"),
            "multiply": ("multiplying", "\\times"),
            "divide": ("dividing", "\\div"),
            "raise": ("raising", "^"),
            "matmul": ("multiplying", "\\dot"),
        }
               
        steps_mathjax.append(
            "Let \\( A \\) and \\( B \\) be two \\( m \\times n \\) matrices "
            "as given below."
        )
        steps_mathjax.append(f"\\[ {self.matrices_a_and_b} \\]")
        
        intro = (
            f"The matrix \\( C \\), obtained by {_map[operation][0]} "
            "the corresponding elements of matrices \\( A \\) and "
            f"\\( {B} \\), is denoted by "
            f"\\( C = A {_map[operation][1]} {B} \\). "
            f"For each element \\( a_{{ij}} \\) of matrix \\( A \\) and each "
            f"element \\( {b_ij} \\) of matrix "
            f"\\( {B} \\), the corresponding element "
            f"\\( c_{{ij}} \\) in the resulting matrix \\( C \\) is "
            f"calculated as \\( c_{{ij}} = a_{{ij}} "
            f"{_map[operation][1]} {b_ij} \\)."
        )
        
        if operation == "subtract":
            old = (
                "subtracting the corresponding elements of matrices "
                f"\\( A \\) and \\( {{\\color{{#01B3D1}}{{B}}}} \\)"
            )
            new = (
                f"subtracting the elements of matrix "
                f"\\( {{\\color{{#01B3D1}}{{B}}}} \\) from the corresponding "
                "elements of matrix \\( A \\) "
            )
            intro = intro.replace(old, new)
            
        if operation == "divide":
            old = (
                "dividing the corresponding elements of matrices \\( A \\) "
                f"and \\( {{\\color{{#01B3D1}}{{B}}}} \\)"
            )
            new = (
                "dividing the elements of matrix \\( A \\) by the "
                "corresponding elements of matrix "
                f"\\( {{\\color{{#01B3D1}}{{B}}}} \\)"
            )
            intro = intro.replace(old, new)
            
        if operation == "raise":
            old = (
                "raising the corresponding elements of matrices \\( A \\) "
                f"and \\( {{\\color{{#01B3D1}}{{B}}}} \\)"
            )
            new = (
                "raising the elements of matrix \\( A \\) to the "
                f"corresponding elements of matrix "
                f"\\( {{\\color{{#01B3D1}}{{B}}}} \\)"
            )
            intro = intro.replace(old, new).replace("^", "\\:^")
            
        steps_mathjax.append(intro)
        
        return steps_mathjax
    
    
    @property
    def b_reciprocal(self) -> str:
        return general_array_ab(
            A="a", B=f"\\frac{{1}}{{b}}", operation="\\div"
        ).replace("\\div", "\\times")
    
    
    @property
    def bg_linalg_add(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self._introduction(operation="add"))
        steps_mathjax.append(
            f"\\[ \\quad C "
            f"= A + {self._B} "
            f"= {general_array_ab()} \\]"
        )
        
        return steps_mathjax


    @property
    def bg_linalg_subtract(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self._introduction(operation="subtract"))
        steps_mathjax.append(
            f"\\[ \\quad C "
            f"= A - {self._B} "
            f"= {general_array_ab(operation='-')} \\]"
        )
        
        return steps_mathjax


    @property
    def bg_linalg_multiply(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self._introduction(operation="multiply"))
        oper = "\\times"
        steps_mathjax.append(
            f"\\[ \\quad C "
            f"= A \\times {self._B} "
            f"= {general_array_ab(operation=oper)} \\]"
        )
        
        return steps_mathjax


    @property
    def bg_linalg_divide(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self._introduction(operation="divide"))
        oper = "\\div"
        steps_mathjax.append(
            f"\\[ \\quad C "
            f"= A \\div {self._B} "
            f"= {general_array_ab(operation=oper)} "
            f"\\equiv {self.b_reciprocal} \\]"
        )
        
        return steps_mathjax


    @property
    def bg_linalg_raise(self) -> list[str]:
    
        steps_mathjax: list[str] = []

        steps_mathjax.extend(self._introduction(operation="raise"))
        steps_mathjax.append(
            f"\\[ \\quad C "
            f"= A \\:^ {self._B} "
            f"= {general_array_ab(operation='^')} \\]"
        )
        
        return steps_mathjax
        
    
    @property
    def bg_matmul(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        dims_m = str_color_math("m")
        dims_p = str_color_math("p", color="red")
        dims_n = str_color_math("n")
        dims_ab = f"{{{dims_m} \\times {dims_n}}}"
        
        a_dims_latex = f"A_{{{dims_m} \\times {dims_p}}}"
        b_dims_latex = f"B_{{{dims_p} \\times {dims_n}}}"
        
        steps_mathjax.append(
            "Let \\( A \\) and \\( B \\) be two \\( m \\times n \\) matrices "
            "as given below."
        )
        steps_mathjax.append(f"\\[ {self.matrices_a_and_b} \\]")
        steps_mathjax.append(
            f"The matrix product \\( {a_dims_latex} \\cdot {b_dims_latex} \\)"
            f"is an \\( {dims_ab} \\) matrix \\( C \\) whose entries are "
            "given by:"
        )
        steps_mathjax.append(
            f"\\( \\quad c_{{ij}} "
            f"= \\sum\\limits_{{k = 1}}^{{m}} \\: "
            f"a_{{ik}} \\: {str_color_math('b_{{kj}}')} "
            f"= a_{{i1}} \\: {str_color_math('b_{{1j}}')} + "
            f"a_{{i2}} \\: {str_color_math('b_{{2j}}')} + "
            f"\\cdots + a_{{im}} \\: {str_color_math('b_{{mj}}')} \\)"
        )
        steps_mathjax.append("That is,")
        steps_mathjax.append(f"\\( \\quad {generate_a_times_b()} \\)")
        
        return steps_mathjax