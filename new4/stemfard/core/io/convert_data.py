from typing import Any
from dataclasses import dataclass
import re

from sympy import Basic


@dataclass
class ConvertData:
    """Typed container for CAS dataset outputs."""
    python: str | None = None
    matlab: str | None = None
    maple: str | None = None
    r: str | None = None
    stata: str | None = None


class DataConverter:
    """
    Convert numeric / symbolic Python data structures into
    Python, Matlab, Maple, R, and Stata datasets.

    - Never evaluates symbolic expressions
    - Preserves exact mathematical structure
    - Handles scalars, vectors, and matrices

    Examples
    --------
    >>> import sympy as sym
    >>> conv = DataConverter()
    """
    def _is_matrix(self, data: Any) -> bool:
        return (
            isinstance(data, (list, tuple))
            and data
            and all(isinstance(r, (list, tuple)) for r in data)
        )
    
    def _map_to_str(self, x: list[Any]) -> list[str]:
        return list(map(str, x))

    def _is_string(self, x: Any) -> bool:
        return isinstance(x, str)

    def _to_float_str(self, x: Any) -> str:
        try:
            return str(float(x))
        except (TypeError, ValueError):
            return str(x)

    def _quote(self, s: str) -> str:
        return f'"{s}"'
    
    def _extract_symbols(self, data: Any) -> set[str]:

        symbols = set()

        def scan(x):
            if isinstance(x, Basic):  # any sympy expression
                symbols.update(str(s) for s in x.free_symbols)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    scan(v)
            elif isinstance(x, str):
                # fallback for plain strings
                for t in re.findall(r"[a-zA-Z_]\w*", x):
                    symbols.add(t)

        scan(data)

        matlab_reserved = {
            "pi", "sqrt", "sin", "cos", "tan", "exp", "log", "log10", "abs"
        }

        return symbols - matlab_reserved
    
    def _contains_symbolic(self, data: Any) -> bool:
        """
        Returns True if `data` contains any SymPy symbolic expressions
        or symbolic variables.
        Works for scalars, lists, and matrices (list of lists).
        """
        def scan(x: Any) -> bool:
            if isinstance(x, Basic):  # SymPy expression
                return bool(x.free_symbols)
            elif isinstance(x, (list, tuple)):
                return any(scan(v) for v in x)
            return False  # plain numbers or strings are not symbolic

        return scan(data)

    # ---------- Backends ----------
    
    def to_python(self, data) -> str:
        return repr(data)

    def to_matlab(self, data: Any) -> str:
        symbolic = self._contains_symbolic(data)

        syms_line = ""
        if symbolic:
            symbols = sorted(self._extract_symbols(data))
            if symbols:
                syms_line = "syms " + " ".join(symbols)

        if self._is_matrix(data):
            rows = [" ".join(self._map_to_str(r)) for r in data]
            body = f"[{'; '.join(rows)}]"

        elif isinstance(data, (list, tuple)):
            body = f"[{' '.join(self._map_to_str(data))}]"

        if syms_line:
            return "\n".join([syms_line, body])

        return body
    
    def to_maple(self, data: Any) -> str:
        if self._is_matrix(data):
            rows = [f"[{', '.join(self._map_to_str(r))}]" for r in data]
            return f"Matrix([{', '.join(rows)}])"

        if isinstance(data, (list, tuple)):
            return f"[{', '.join(self._map_to_str(data))}]"

        return str(data)

    def to_r(self, data: Any, allow_symbolic: bool = False ) -> str | None:
        # Return None if any value is symbolic,
        # i.e. we are not implimenting symbolic operations in R
        if not allow_symbolic and self._contains_symbolic(data):
            return None

        if self._is_matrix(data):
            flat = [str(v) for row in data for v in row]
            vec = f"x <- c({', '.join(flat)})"
            mat = f"M <- matrix(x, nrow={len(data)}, byrow=TRUE)"
            return "\n".join([vec, mat])

        if isinstance(data, (list, tuple)):
            return f"c({', '.join(map(str, data))})"

        return str(data)

    def to_stata(self, data: Any) -> str:
        
        if self._is_matrix(data):
            columns = list(zip(*data))
            colnames = []

            # infer column types and widths
            colinfo = []
            for i, col in enumerate(columns, start=1):
                if any(self._is_string(v) for v in col):
                    maxlen = max(len(str(v)) for v in col)
                    colnames.append(f"var{i}str{maxlen}")
                    colinfo.append("string")
                else:
                    colnames.append(f"var{i}")
                    colinfo.append("numeric")

            header = "input " + " ".join(colnames)

            rows = []
            for row in data:
                out = ["\t"]
                for v, kind in zip(row, colinfo):
                    if kind == "string":
                        out.append(self._quote(str(v)))
                    else:
                        out.append(self._to_float_str(v))
                rows.append("\t".join(out))
                
            return (
                "\n".join(["clear", header] + rows + ["end"]).replace("\t\t", "\t")
            )

        if isinstance(data, (list, tuple)):
            if any(self._is_string(v) for v in data):
                maxlen = max(len(str(v)) for v in data)
                header = f"input str{maxlen} var1"
                rows = [self._quote(str(v)) for v in data]
            else:
                header = "input var1"
                rows = [self._to_float_str(v) for v in data]
                
            rows = [f"\t{row}" for row in rows]

            return "\n".join(["clear", header] + rows + ["end"])

        return self._to_float_str(data)