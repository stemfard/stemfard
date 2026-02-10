class OperationError(ValueError):
    """
    Raised when a specified operation is invalid.
    """
    def __init__(
        self,
        operation: str,
        valid_operations: tuple[str, ...],
        message: str | None = None
    ):
        
        if message is None:
            message = (
                f"Expected 'operation' to be one of: "
                f"{', '.join(valid_operations)}; got {operation!r}"
            )
        super().__init__(message)