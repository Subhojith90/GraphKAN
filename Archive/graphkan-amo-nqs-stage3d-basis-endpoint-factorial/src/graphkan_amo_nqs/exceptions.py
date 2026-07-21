class GraphKANError(Exception):
    """Base exception for the package."""


class DependencyUnavailable(GraphKANError):
    """Raised when an optional dependency is required but unavailable."""


class SchemaError(GraphKANError):
    """Raised when an output schema is invalid."""


class PhysicsValidationError(GraphKANError):
    """Raised when a physics validation check fails."""
