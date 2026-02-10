from .interpolate import Interpolate
from .dispatcher import InterpDispatcher

# import modules to populate METHODS = {} dictionary with registered interpolation method
from .methods import lagrange, hermite, polynomial, quadratic, saturation

__all__ = ["Interpolate", "InterpDispatcher"]


# IVPSolver()
# Interpolator1D
# NonlinearSolver
# IterativeLinearSolver
# DirectLinearSolver