class AffineError(Exception):
    """Base exception for the affine tool."""
    pass

class ConfigurationError(AffineError):
    """Raised for configuration-related errors."""
    pass

class LLMAPIError(AffineError):
    """Raised for errors related to the LLM API."""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code 