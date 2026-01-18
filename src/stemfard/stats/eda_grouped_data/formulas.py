"""
Formulas and constants for grouped statistics.
"""

# Constants
ALLOWED_STATISTICS = ["mean", "std", "percentiles"]
ALLOWED_FORMULAS = ["x-a", "x/w-a", "(x-a)/w"]
FIGURE_COLORS = ["blue", "black", "green", "red", "purple", "orange"]
TABLE_CALC_INTRO = "The table below is generated for ease of calculations"

# Formula definitions
FORMULAS_ARITHMETIC_MEAN: dict[str, str] = {
    "values_only": (
        f"x_{{1}}, \\: x_{{2}}, \\: x_{{3}}, \\: \\cdots, \\: x_{{n}}"
    ),
    "values_and_freq": (
        f"\\begin{{array}}{{|l|c|c|c|c|c|}} \\hline "
        f"\\mathrm{{Data}} & x_{{1}} & x_{{2}} & x_{{3}} & \\cdots & x_{{n}} \\\\ \\hline "
        f"\\mathrm{{Frequency}} & f_{{1}} & f_{{2}} & f_{{3}} & \\cdots & f_{{n}} \\\\ \\hline "
        f"\\end{{array}}"
    ),
    "arithmetic_mean": (
        f"\\displaystyle \\bar{{x}} = \\frac{{ \\sum x_{{i}}}}{{n}}"
    ),
    "arithmetic_mean_with_assumed_mean": (
        f"\\displaystyle \\bar{{x}} = A + \\frac{{ \\sum (x_{{i}} - A)}}{{n}}"
    ),
    "arithmetic_mean_with_freq": (
        f"\\displaystyle \\bar{{x}} "
        f"= \\frac{{ \\sum\\mathrm{{fx}}}}{{ \\sum\\mathrm{{f}}}}"
    ),
    "arithmetic_mean_with_freq_and_assumed_mean": (
        f"\\displaystyle \\bar{{x}} "
        f"= A + \\frac{{ \\sum\\mathrm{{ft}}}}{{ \\sum\\mathrm{{f}}}}"
    )
}

FORMULAS_ASSUMED_MEAN_TYPES: dict[str, list[str]] = {
    "x-a": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\)",
        "Where \\( f \\) is the frequency and \\( t = x - A \\)."
    ],
    "x/w-a": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= \\left(A^{{*}} + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\right) "
        "\\times w \\)",
        f"Where \\( A^{{*}} \\) is the new assumed mean, \\( f \\) is the "
        "frequency, \\( t = x - A\\) and \\( w \\) is the "
        f"\\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ],
    "(x-a)/w": [
        f"\\( \\displaystyle \\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\times w \\)",
        "Where \\( f \\) is the frequency, \\( t = x - A\\) and \\( w \\) "
        f"is the \\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ]
}