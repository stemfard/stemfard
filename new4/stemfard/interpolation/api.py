from .request import InterpRequest
from .validated_request import ValidatedInterpRequest
from .dispatcher import InterpDispatcher

def dispatch_interpolate(**kwargs):
    """
    Public API to perform interpolation using dispatcher.
    Example:
        dispatch_interpolate(x=[1,2], y=[3,4], x0=1.5, method='lagrange')
        
    NOTES
    This is the only entry point.
    Handles request creation, validation, and dispatch.
    """
    raw_req = InterpRequest(**kwargs)
    validated_req = ValidatedInterpRequest(raw_req)
    return InterpDispatcher.dispatch(validated_req)
