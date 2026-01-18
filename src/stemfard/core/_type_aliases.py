from numpy import float64
from numpy.typing import NDArray

ArrayLike = list[int | float] | NDArray[float64]
ScalarArrayLike = int | float | list[int | float] | NDArray