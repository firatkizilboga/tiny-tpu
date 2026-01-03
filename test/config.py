"""
Central configuration for TinyTPU tests.
Array size is read from TPU_ARRAY_SIZE environment variable.
"""

import os

# Read array size from environment, default to 2
ARRAY_SIZE = int(os.environ.get('TPU_ARRAY_SIZE', '2'))

# Validate array size
if ARRAY_SIZE < 2:
    raise ValueError(f"ARRAY_SIZE must be >= 2, got {ARRAY_SIZE}")

def get_array_size():
    """Return the configured array size."""
    return ARRAY_SIZE
