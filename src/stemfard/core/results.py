from typing import Any

class Result:
    """
    Generic result container for any function.

    - Attributes are created dynamically based on what the function returns.
    - Supports dot access: res.attr
    - Provides .keys(), .items(), and a summary method.
    """

    def __init__(self, **kwargs: Any):
        # Add all items as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def keys(self):
        """Return all attribute names."""
        return list(self.__dict__.keys())

    def items(self):
        """Return all attribute-name / value pairs."""
        return self.__dict__.items()

    def summary(self):
        """Pretty-print all stored attributes."""
        for key, value in self.items():
            print(f"{key}: {value}")

    def __repr__(self) -> str:
        attrs = ", ".join(self.keys())
        return f"Result({attrs})"