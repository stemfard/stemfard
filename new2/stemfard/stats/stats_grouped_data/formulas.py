"""
Formulas and constants for grouped statistics.
"""

# Constants
ALLOWED_STATISTICS: list[str] = ["mean", "std", "percentiles"]
ALLOWED_FORMULAS: list[str] = ["x-a", "x/w-a", "(x-a)/w"]
FIGURE_COLORS: list[str] = ["blue", "black", "green", "red", "purple", "orange"]
TABLE_CALC_INTRO: str = "The table below is generated for ease of calculations"

FORMULAS_ASSUMED_MEAN_TYPES: dict[str, list[str]] = {
    "x-a": [
        f"\\( \\displaystyle\\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\)",
        "Where \\( f \\) is the frequency and \\( t = x - A \\)."
    ],
    "x/w-a": [
        f"\\( \\displaystyle\\quad \\bar{{x}} "
        f"= \\left(A^{{*}} + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\right) "
        "\\times w \\)",
        f"Where \\( A^{{*}} \\) is the new assumed mean, \\( f \\) is the "
        "frequency, \\( t = x - A\\) and \\( w \\) is the "
        f"\\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ],
    "(x-a)/w": [
        f"\\( \\displaystyle\\quad \\bar{{x}} "
        f"= A + \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\times w \\)",
        "Where \\( f \\) is the frequency, \\( t = x - A\\) and \\( w \\) "
        f"is the \\( \\textbf{{class width}} \\)CLASS_WIDTH"
    ]
}