"""In this example we compute embedding and run masked inference
 on the ESM2 language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    esm_mean_embedding_params,
    esm_masked_inference_params,
)

client = GinkgoAIClient()

# Simple query for embedding computation
prediction = client.query(esm_mean_embedding_params("MLYLRRL"))
# prediction["embedding"] == [1.05, -2.34, ...]


# Simple query for masked inference
prediction = client.query(esm_masked_inference_params("MLY<mask>RRL"))

queries = [
    esm_mean_embedding_params("MLYLRRL"),
    esm_mean_embedding_params("MLYRRL"),
    esm_mean_embedding_params("MLYLLRRL"),
]
predictions = client.batch_query(queries)

# predictions[0]["result"]["embedding"] == [1.05, -2.34, ...]
