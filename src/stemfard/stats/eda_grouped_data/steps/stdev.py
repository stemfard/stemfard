from typing import Literal

from numpy import float64, int64
from numpy.typing import NDArray
from verifyparams import verify_membership

def sta_eda_grouped_std_steps(
    class_width: int | float,
    midpoints: NDArray[float64],
    freq: NDArray[int64],
    assumed_mean: int | float,
    formula: Literal["x-a", "x/w-a", "(x-a)/w"],
    formula_std: Literal[1, 2, 3] = 1
) -> float:
    """
    formula_std: {1, 2, 3}, default None
        Standard deviation formula
            `1` - `sum(f * d^2) / sum(f)` where d = x - mean(x)
            `2` - `sum(f * x^2) / sum(f) - mean(x)^2`. This is 
                  the expanded form of `1`.
            `3` - `c^2 * [sum(f * t^2) / sum(f) - mean(t)^2]
                  where `t = (x - A) / c` with `A` and `c` as the
                  assumed mean and class width respectively.
    """
    x = midpoints
    f = freq
    w = class_width
    
    verify_membership(
        value=formula_std,
        valid_items=[1, 2, 3],
        param_name="formula_std"
    )
        
    if formula_std == 1:
        d = x - x.mean()
        return (f * d ** 2).sum() / f.sum()
    elif formula == 2:
        return (f * x ** 2).sum() / f.sum() - x.mean() ** 2
    else: # 3
        A = assumed_mean
        if formula == "x-a":
            mean_of_t = (f * t).sum() / f.sum()
        elif formula == "x/w-a":
            t = x / w - A / w
            mean_of_t = (f * t).sum() / f.sum()
        else: # None | "(x-a)/w"
            t = (x - A) / w
            mean_of_t = (f * t).sum() / f.sum()

        return w ** 2 * ( (f * t ** 2).sum() / f.sum() - mean_of_t ** 2 )