import pandas as pd
from sympy import lambdify, symbols

def make_table(req, poly, derivative=None):
    """
    Create a table of x, f(x), and optionally derivatives.
    """
    x_sym = symbols("x")
    f = lambdify(x_sym, poly, "numpy")

    data = {"x": req.x, "f(x)": [float(f(xi)) for xi in req.x]}

    if derivative:
        df = lambdify(x_sym, derivative, "numpy")
        data["f'"] = [float(df(xi)) for xi in req.x]

    return pd.DataFrame(data)


import matplotlib.pyplot as plt
import numpy as np

def make_plot(req, poly, derivative=None, x0=None, plot_deriv=False):
    """
    Plot the interpolation polynomial and optionally its derivative.
    """
    x_sym = symbols("x")
    f = lambdify(x_sym, poly, "numpy")

    x_vals = np.linspace(min(req.x), max(req.x), 500)
    y_vals = f(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="Interpolated Poly", color="blue")
    ax.scatter(req.x, req.y, color="red", label="Data points")

    # Plot derivative if requested
    if derivative is not None and plot_deriv:
        df = lambdify(x_sym, derivative, "numpy")
        y_der = df(x_vals)
        ax.plot(x_vals, y_der, "--", label=f"{req.expr_variable}'", color="green")

    # Highlight x0
    if x0 is not None:
        ax.scatter([x0], [float(f(x0))], color="orange", zorder=5, label=f"{req.expr_variable}_0")

    ax.set_xlabel(req.expr_variable)
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    return fig