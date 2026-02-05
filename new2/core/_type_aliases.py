from typing import Sequence
from numpy import float64, int64
from numpy.typing import NDArray
from sympy import Matrix

SequenceArrayLike = Sequence[int | float] | NDArray[int64 | float64]
Array2DLike = NDArray[float64]
Matrix2DLike = Matrix
Array2DMatrixLike = list[list[int | float | str]] | NDArray | Matrix
ScalarSequenceArrayLike = int | float | Sequence[int | float] | NDArray[int64 | float64]
IntegerSequenceArrayLike = int | Sequence[int] | NDArray[int64]