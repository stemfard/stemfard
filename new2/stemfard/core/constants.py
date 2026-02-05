from stemfard.core._html import html_border, html_border_dashed, html_border_dotted, html_border_solid


HEXCOLORS = [
    "0000FF", "8A2BE2", "FF8C00", "00BFFF", "6495ED", "00008B", "008B8B",
    "008000", "8B0000", "9ACD32", "B8860B", "BDB76B", "778899", "008080",
    "000000"
]

class StemConstants:
    """Class with constants"""
    
    FASTAPI_HOST_URL = "https://localhost:8000"
    CHECKMARK = f"\\:{{\\color{{red}}{{\\checkmark}}}}"
    SPEECH_FINAL = "THIS IS THE FINAL ANSWER."
    BORDER_HTML_SOLID = html_border_solid()
    BORDER_HTML_DASHED = html_border_dashed()
    BORDER_HTML_DOTTED = html_border_dotted()
    BORDER_HTML_BLUE_WIDTH_2 = html_border(type="solid", width=2, color="skyblue")
    BORDER_LATEX = "\\hlinecustom[dotted]{0.5pt}{gray!60}"
    HEXCOLORS = HEXCOLORS

    STEPS_NULL = {
        "steps_mathjax": None,
        "others": {
            "remarks": None,
            "warnings": None
        }
    }

    STEPS_NOT_REQUESTED = {
        "steps_mathjax": [],
        "others": {
            "remarks": [],
            "warnings": []
        }
    }