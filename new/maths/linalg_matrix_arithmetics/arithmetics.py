from stemfard.core._type_aliases import ScalarSequenceArrayLike, SequenceArrayLike
from stemfard.maths.linalg_matrix_arithmetics.core import MatrixArithmetics


def linalg_add(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_add()


def linalg_subtract(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_subtract()


def linalg_multiply(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_multiply()


def linalg_divide(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_divide()


def linalg_raise(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_raise()


def linalg_add_scalar(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_add_scalar()


def linalg_subtract_scalar(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_subtract_scalar()


def linalg_multiply_scalar(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_multiply_scalar()


def linalg_divide_scalar(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_divide_scalar()


def linalg_raise_scalar(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_linalg_raise_scalar()


def linalg_matmul(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_names: tuple[str, str] = ("A", "B"),
    decimals: int = 12
) -> MatrixArithmetics:
    
    params = {
        "method": "add",
        "A": A,
        "B": B,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_names": param_names,
        "decimals": decimals
    }
    
    return MatrixArithmetics(**params).mth_linalg_matmul()