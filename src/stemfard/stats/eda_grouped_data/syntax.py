from typing import Literal

from numpy import float64
from numpy.typing import NDArray

from stemfard.core.decimals import numeric_format
from stemfard.core.utils_classes import ResultDict


def _syntax_mean(formula: Literal["x-a", "x/w-a", "(x-a)/w"]) -> list[str]:
    
    matlab = ""
    numpy = ""
    scipy = None
    statsmodels = None
    sympy = None
    r = ""
    stata = None
    
    temp_str = (
        "# perform calculations\n"
        "x = (lower_limits + upper_limits) / 2 # midpoints (x)\n"
    )
    
    matlab += temp_str.replace(" #", "; %")
    numpy += temp_str
    r += temp_str
    
    if formula is None:
        matlab += (
            "fx = f .* x;\n"
            "mean = sum(fx) / sum(f); % total fx / total f\n"
            "round(mean, decimals)"
        )
        numpy += (
            "fx = f * x\n"
            "mean = fx.sum() / f.sum() # total fx / total f\n"
            "round(mean, decimals)"
        )
        r += (
            "fx = f * x;\n"
            "mean = sum(fx) / sum(f) # total fx / total f\n"
            "round(mean, decimals)"
        )
    elif formula == "x-a":
        matlab += (
            f"t = x - A;\n"
            f"ft = f .* t;\n"
            "mean = A + sum(ft) / sum(f); % total ft / total f\n"
            "round(mean, decimals)"
        )
        numpy += (
            f"t = x - A \n"
            f"ft = f * t\n"
            "mean = A + ft.sum() / f.sum() # total ft / total f\n"
            "round(mean, decimals)"
        )
        r += (
            f"t = x - A \n"
            f"ft = f * t\n"
            "mean = A + sum(ft) / sum(f) # total ft / total f\n"
            "round(mean, decimals)"
        )
    elif formula == "x/w-a":
        matlab += (
            f"w = upper_limits(1) - lower_limits(1) + 1; % class width\n"
            f"t = x ./ w - A;\n"
            f"ft = f .* t;\n"
            "mean = (A + sum(ft) / sum(f)) * w;\n"
            "round(mean, decimals)"
        )
        numpy += (
            "w = upper_limits[0] - lower_limits[0] + 1 # class width\n"
            f"t = x / w - A \n"
            f"ft = f * t\n"
            "mean = (A + ft.sum() / f.sum()) * w\n"
            "round(mean, decimals)"
        )
        r += (
            "w = upper_limits[1] - lower_limits[1] + 1 # class width\n"
            f"t = x / w - A \n"
            f"ft = f * t\n"
            "mean = (A + sum(ft) / sum(f)) * w\n"
            "round(mean, decimals)"
        )
    elif formula == "(x-a)/w":
        matlab += (
            f"w = upper_limits(1) - lower_limits(1) + 1; % class width\n"
            f"t = (x - A) / w;\n"
            f"ft = f .* t;\n"
            "mean = A + sum(ft) / sum(f) * w;\n"
            "round(mean, decimals)"
        )
        numpy += (
            "w = upper_limits[0] - lower_limits[0] + 1 # class width\n"
            f"t = (x - A) / w \n"
            f"ft = f * t\n"
            "mean = A + ft.sum() / f.sum() * w\n"
            "round(mean, decimals)"
        )
        r += (
            "w = upper_limits[1] - lower_limits[1] + 1 # class width\n"
            f"t = (x - A) / w \n"
            f"ft = f * t\n"
            "mean = A + sum(ft) / sum(f) * w\n"
            "round(mean, decimals)"
        )
        
    return ResultDict(
        matlab=matlab,
        numpy=numpy,
        scipy=scipy,
        statsmodels=statsmodels,
        sympy=sympy,
        r=r,
        stata=stata
    )
    
    
def _syntax_std(
    formula: Literal["x-a", "x/w-a", "(x-a)/w"],
    formula_std: Literal[1, 2, 3]
) -> list[str]:
    
    result = _syntax_mean(formula)
    
    matlab=result.matlab
    numpy=result.numpy
    scipy=result.scipy
    statsmodels=result.statsmodels
    sympy=result.sympy
    r=result.r
    stata=result.stata
    
    if formula_std == 1:
        matlab += "XXX"
        numpy += "XXX"
        r += "XXX"
    elif formula_std == 2:
        matlab += "XXX"
        numpy += "XXX"
        r += "XXX"
    else:
        matlab += "XXX"
        numpy += "XXX"
        r += "XXX"
        
    return ResultDict(
        matlab=matlab,
        numpy=numpy,
        scipy=scipy,
        statsmodels=statsmodels,
        sympy=sympy,
        r=r,
        stata=stata
    )


def _syntax_percentiles() -> list[str]:
    
    matlab = ""
    numpy = ""
    scipy = None
    statsmodels = None
    sympy = None
    r = ""
    stata = None
        
    return ResultDict(
        matlab=matlab,
        numpy=numpy,
        scipy=scipy,
        statsmodels=statsmodels,
        sympy=sympy,
        r=r,
        stata=stata
    )


def syntaxes(
    lower: NDArray[float64],
    upper: NDArray[float64],
    class_width: int | float,
    assumed_mean: int | float,
    assumed_mean_asteriks_rounded: float,
    formula: Literal["x-a", "x/w-a", "(x-a)/w"],
    statistic: Literal["mean", "std", "percentiles"],
    decimals: int
) -> list[str]:
        """
        Syntax
        """
        matlab = ""
        numpy = ""
        r = ""
        
        lower = f"lower = {lower}"
        upper = f"upper = {upper}"
        freq = f"f = {freq}"
        
        if assumed_mean is not None:
            if formula == "x/w-a":
                assumed_mean_str = (
                    f"A = {numeric_format(assumed_mean)} / "
                    f"{numeric_format(class_width)} "
                    f"# i.e. assumed_mean / class_width "
                    f"= {assumed_mean_asteriks_rounded}"
                )
            else:
                assumed_mean_str = f"A = {assumed_mean} # assumed_mean"
        else:
            assumed_mean_str = ""
        
        # Matlab
        # ------
        
        matlab += f"% define inputs\n{lower};\n{upper};\n{freq};\n"
        if assumed_mean_str:
            matlab += (
                f"{assumed_mean_str.replace(' #', '; %')}\n"
                f"decimals = {decimals};\n\n"
            )
        else:
            matlab += f"decimals = {decimals};\n\n"
        
        # Numpy
        # -----
        numpy += (
            "import numpy as np\n\n"
            "# define inputs\n"
            f"{lower.replace('[', 'np.array([')})\n"
            f"{upper.replace('[', 'np.array([')})\n"
            f"{freq.replace('[', 'np.array([')})\n"
        )
            
        if assumed_mean_str:
            numpy += f"{assumed_mean_str}\ndecimals = {decimals} \n\n"
        else:
            numpy += f"decimals = {decimals} \n\n"
        
        # r
        # -
        r += f"# define inputs\n{lower}\n{upper}\n{freq}\n"\
            .replace("[", "c(").replace("]", ")")
        
        if assumed_mean_str:
            r += f"{assumed_mean_str}\ndecimals = {decimals} \n\n"
        else:
            r += f"decimals = {decimals} \n\n"
        
        if statistic == "mean":
            syntax = _syntax_mean()
        elif statistic == "sd":
            syntax = _syntax_std()
        elif statistic == "percentiles":
            syntax = _syntax_percentiles()
        
        return ResultDict(
            matlab=syntax.matlab,
            numpy=syntax.numpy,
            scipy=syntax.scipy,
            statsmodels=syntax.statsmodels,
            sympy=syntax.sympy,
            r=syntax.r,
            stata=syntax.stata
        )