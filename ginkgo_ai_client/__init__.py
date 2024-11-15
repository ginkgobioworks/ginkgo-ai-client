__version__ = "0.0.3"

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
