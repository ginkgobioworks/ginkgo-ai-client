__version__ = "0.2.0"

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
