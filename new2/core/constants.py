from stemfard.core._html import html_border


class StemConstants:
    """Class with constants"""
    
    FASTAPI_HOST_URL = "https://localhost:8000"
    CHECKMARK = f"\\:{{\\color{{red}}{{\\checkmark}}}}"
    SPEECH_FINAL = "THIS IS THE FINAL ANSWER."
    BORDER_HTML = html_border()
    BORDER_HTML_BG = html_border(type="solid", width=2, color="skyblue")
    BORDER_LATEX = "\\hlinecustom[dotted]{0.5pt}{gray!60}"

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