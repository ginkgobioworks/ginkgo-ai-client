"""In this example we compute embedding and run masked inference
 on Ginkgo's AA0 protein language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    aa0_mean_embedding_params,
    aa0_masked_inference_params,
)

client = GinkgoAIClient()

# Simple query for embedding computation
prediction = client.query(aa0_mean_embedding_params("ATTGCG"))
# prediction["embedding"] == [1.05, -2.34, ...]


# Simple query for masked inference
prediction = client.query(aa0_masked_inference_params("ATT<mask>TAC"))

queries = [
    aa0_mean_embedding_params("AGCGC"),
    aa0_mean_embedding_params("ATTGCG"),
    aa0_mean_embedding_params("TACCGCA"),
]
predictions = client.batch_query(queries)

# predictions[0]["result"]["embedding"] == [1.05, -2.34, ...]
