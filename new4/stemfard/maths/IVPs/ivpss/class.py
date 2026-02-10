from typing import Optional, Union
from pandas import DataFrame, Styler
from sympy import Expr
from matplotlib.figure import Figure

class IVPResult:
    """
    Container for IVP solver results.
    Stores:
      - table of times, approximations, exact values, errors
      - styled table
      - final value
      - plots (optional)
    """
    def __init__(
        self,
        table: Optional[DataFrame] = None,
        dframe: Optional[Styler] = None,
        answer: Optional[float] = None,
        plot: Optional[Figure] = None
    ):
        self.table = table
        self.dframe = dframe
        self.answer = answer
        self.plot = plot

    def __repr__(self):
        if self.table is not None:
            return repr(self.table)
        return "<IVP Result>"

    def _repr_html_(self):
        if self.dframe is not None:
            return self.dframe._repr_html_()
        return "<IVP Result>"
