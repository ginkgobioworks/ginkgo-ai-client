"""This runs embedding and masked inference queries on Ginkgo's 3'UTR language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    aa0_mean_embedding_params,
    aa0_masked_inference_params,
)

client = GinkgoAIClient()

# Simple query for embedding computation
prediction = client.query(aa0_mean_embedding_params("MLYLRRL"))
# prediction["embedding"] == [1.05, -2.34, ...]


# Simple query for masked inference
prediction = client.query(aa0_masked_inference_params("MLY<mask>RRL"))

queries = [
    aa0_mean_embedding_params("MLYLRRL"),
    aa0_mean_embedding_params("MLYRRL"),
    aa0_mean_embedding_params("MLYLLRRL"),
]
predictions = client.batch_query(queries)

# predictions[0]["result"]["embedding"] == [1.05, -2.34, ...]
