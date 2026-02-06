import numpy as np
from .request import InterpRequest

class ValidatedInterpRequest:
    ALLOWED_METHODS = {
        'straight-line', 'lagrange', 'hermite', 'newton-backward',
        'newton-forward', 'gauss-backward', 'gauss-forward',
        'newton-divided', 'neville', 'stirling', 'bessel', 'laplace-everett',
        'linear-splines', 'quadratic-splines', 'natural-cubic-splines',
        'clamped-cubic-splines', 'not-a-knot-splines',
        'linear-regression', 'polynomial', 'exponential', 'power',
        'saturation', 'reciprocal'
    }

    def __init__(self, req: InterpRequest):
        self.req = req
        self.x = np.asarray(req.x, dtype=float)
        self.y = np.asarray(req.y, dtype=float)
        self.x0 = req.x0
        self.method = req.method
        self.expr_variable = req.expr_variable
        self.decimals = req.decimals
        self.symbolic = req.symbolic
        self.poly_order = req.poly_order
        self.qs_constraint = req.qs_constraint
        self.sat_type = req.sat_type

        # Optional derivative for Hermite
        self.yprime = None
        if self.method == 'hermite':
            if req.yprime is None:
                raise ValueError("Hermite interpolation requires yprime")
            self.yprime = np.asarray(req.yprime, dtype=float)
            if self.yprime.shape != self.y.shape:
                raise ValueError("yprime must have same length as y")

        # Validate main constraints
        self._validate()

    def _validate(self):
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have same length")
        if self.method not in self.ALLOWED_METHODS:
            raise ValueError(f"Unknown interpolation method '{self.method}'")
        if self.method in {'polynomial', 'linear-regression'} and self.poly_order is None:
            raise ValueError(f"'{self.method}' requires poly_order to be specified")
        if self.method == 'saturation' and self.sat_type not in {'ax/(x+b)', 'x/(ax+b)'}:
            raise ValueError(f"Invalid sat_type '{self.sat_type}'")