class InsufficientStockError(Exception):
    """Raised when more units are requested than are in stock."""
