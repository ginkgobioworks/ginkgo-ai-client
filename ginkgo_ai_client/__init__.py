__version__ = "0.1.0"

from .client import GinkgoAIClient

from .queries import (
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
)

__all__ = [
    "GinkgoAIClient",
    "MaskedInferenceQuery",
    "MeanEmbeddingQuery",
]
