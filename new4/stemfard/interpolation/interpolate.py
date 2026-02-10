from pandas import DataFrame
from numpy import array, asarray
from numpy.typing import NDArray

from stemfard.interpolation.api import dispatch_interpolate


class Interpolate:
    """
    User-facing interpolation container.

    - Holds data (x, y)
    - Displays tables nicely
    - Delegates computation to dispatcher
    """

    def __init__(self, x, y, index=None, columns=None, decimals: int = 12):
        self.x = asarray(x)
        self.y = asarray(y)
        self.decimals = decimals

        self._df = DataFrame(
            data=array([self.x, self.y]).T.round(decimals),
            index=index,
            columns=columns or ["x", "y"],
        )

    # ---------------------------
    # Display & protocol support
    # ---------------------------

    def __repr__(self):
        return repr(self._df)

    def _repr_html_(self):
        return self._df._repr_html_()

    def _repr_latex_(self):
        latex = self._df.to_latex(index=False)
        return f"$ {latex} $"

    def __len__(self):
        return len(self._df)

    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        self._df[key] = value

    def __iter__(self):
        return iter(self._df)

    @property
    def values(self) -> NDArray:
        return self._df.to_numpy()

    @property
    def shape(self):
        return self._df.shape

    @property
    def points(self):
        return self._df.values.tolist()

    # ---------------------------
    # Interpolation methods
    # ---------------------------

    def linear(self, x0):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            x0=x0,
            method="linear",
        )

    def lagrange(self, x0, symbolic: bool = False):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            x0=x0,
            method="lagrange",
            symbolic=symbolic,
        )

    def hermite(self, x0, yprime, symbolic: bool = False):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            yprime=yprime,
            x0=x0,
            method="hermite",
            symbolic=symbolic,
        )

    def polynomial(self, x0, order: int = 1, symbolic: bool = False):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            x0=x0,
            method="polynomial",
            poly_order=order,
            symbolic=symbolic,
        )

    def quadratic_spline(self, x0, constraint: float = 0.0):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            x0=x0,
            method="quadratic-splines",
            qs_constraint=constraint,
        )

    def saturation(self, x0, sat_type: str = "ax/(x+b)"):
        return dispatch_interpolate(
            x=self.x,
            y=self.y,
            x0=x0,
            method="saturation",
            sat_type=sat_type,
        )