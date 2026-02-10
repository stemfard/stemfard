from pandas import DataFrame
from numpy import array, asarray, float64, interp
from numpy.typing import NDArray
from sympy import interpolate, lambdify, simplify, symbols, sympify

class Interpolate:
    def __init__(self, x, y, index=None, columns=None, decimals: int = 12):
        """Convert input to pandas DataFrame"""
        self.x = asarray(x)
        self.y = asarray(y)
        self.arr = array([x, y]).T
        self.index = index
        self.columns = columns
        self.decimals = decimals
        
        self._df = DataFrame(
            data=self.arr.round(self.decimals),
            index=self.index,
            columns=self.columns
        )
        
    def __repr__(self):
        """Show DataFrame in terminal"""
        return repr(self._df)

    def _repr_html_(self):
        """Show DataFrame in Jupyter"""
        return self._df._repr_html_()

    def _repr_latex_(self):
        latex = self._df.to_latex(index=False)
        return f"$ {latex} $"
    
    # ---------------------------
    # Python protocol support
    # ---------------------------

    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self._df)

    def __getitem__(self, key):
        """Access data using indexing syntax."""
        return self._df[key]
    
    def __setitem__(self, key, value):
        """Add or modify columns, ie. enable interp["new_col"] = values"""
        self._df[key] = value

    def __iter__(self):
        """Iterate over rows."""
        return iter(self._df)
    
    @property
    def values(self) -> NDArray[float64]:
        return self._df.to_numpy()
    
    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape
    
    @property
    def points(self) -> list[list[float]]:
        """Return data as list of (x, y) tuples."""
        return self._df.values.tolist()
    
    def linear(self):
        """
        Linear interpolation.

        Returns
        -------
        callable
            Function f(x_new) performing linear interpolation.
        """
        x, y = self.x, self.y

        def f(x_new):
            return interp(x_new, x, y)

        return f

    def lagrange(self, symbolic: bool = False):
        """
        Lagrange polynomial interpolation.

        Parameters
        ----------
        symbolic : bool, default False
            If True, return a SymPy expression.
            If False, return a numeric callable.

        Returns
        -------
        sympy.Expr or callable
        """
        x_sym = symbols("x")
        points = list(zip(self.x, self.y))

        poly = interpolate(points, x_sym)

        if symbolic:
            return simplify(poly)

        f_num = lambdify(x_sym, poly, modules="numpy")
        return f_num