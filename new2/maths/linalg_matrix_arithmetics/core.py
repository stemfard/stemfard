from stemfard.maths.linalg_matrix_arithmetics._basex import BaseLinalgMatrixArithmetics
from stemfard.core._latex import tex_to_latex


class MatrixArithmeticsBackground(BaseLinalgMatrixArithmetics):
    """Background"""
    
    @property
    def bg_linalg_add(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        return steps_mathjax


    @property
    def bg_linalg_subtract(self) -> list[str]:
        pass


    @property
    def bg_linalg_multiply(self) -> list[str]:
        pass


    @property
    def bg_linalg_divide(self) -> list[str]:
        pass


    @property
    def bg_linalg_raise(self) -> list[str]:
        pass


    @property
    def bg_linalg_add_scalar(self) -> list[str]:
        pass


    @property
    def bg_linalg_subtract_scalar(self) -> list[str]:
        pass

    
    @property
    def bg_linalg_multiply_scalar(self) -> list[str]:
        pass

    
    @property
    def bg_linalg_divide_scalar(self) -> list[str]:
        pass
    
    
    @property
    def bg_linalg_raise_scalar(self) -> list[str]:
        pass
    
    
    @property
    def bg_matmul(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        steps_mathjax.append(f"To update bg matrix multiplication")
        
        return steps_mathjax


class MatrixArithmetics(MatrixArithmeticsBackground):
    """Matrix arithmetics"""
    
    @property
    def mth_linalg_add(self) -> list[str]:
    
        steps_mathjax: list[str] = []
        
        return steps_mathjax


    @property
    def mth_linalg_subtract(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_multiply(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_divide(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_raise(self) -> list[str]:
        pass
    
    
    @property
    def mth_linalg_matmul(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        if self.prm_show_bg:
            steps_mathjax.extend(self.bg_matmul)
        
        steps_mathjax.append(f"\\( {tex_to_latex(self.AB_rnd)} \\)")
        
        return steps_mathjax


class MatrixArithmeticsScalars(MatrixArithmeticsBackground):

    @property
    def mth_linalg_add_scalar(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_subtract_scalar(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_multiply_scalar(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_divide_scalar(self) -> list[str]:
        pass

    
    @property
    def mth_linalg_raise_scalar(self) -> list[str]:
        pass