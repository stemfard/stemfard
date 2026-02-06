import numpy as np

from stemfard.interpolation.core.utils import make_plot, make_table
from stemfard.interpolation.dispatcher import InterpDispatcher
from stemfard.interpolation.result import InterpResult

@InterpDispatcher.register("quadratic-splines")
def quadratic_spline(req):
    x = np.array(req.x)
    y = np.array(req.y)
    n = len(x) - 1
    a = y[:-1]
    b = (y[1:] - y[:-1]) / (x[1:] - x[:-1]) - req.qs_constraint
    c = req.qs_constraint

    # simple quadratic spline: f_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2
    def f(x0):
        for i in range(n):
            if x[i] <= x0 <= x[i+1]:
                return float(a[i] + b[i]*(x0 - x[i]) + c*(x0 - x[i])**2)
        return None

    fx = f(req.x0)

    # For demonstration, store piecewise polynomial as string
    poly = "Quadratic splines (piecewise)"

    table = make_table(req, poly=None)  # Table can be just x & y
    plot = None  # Could implement piecewise plot if needed

    return InterpResult(
        table=table,
        polynomial=poly,
        value_at_x0=fx,
        plot=plot
    )
