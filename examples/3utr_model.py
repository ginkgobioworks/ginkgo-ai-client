"""This runs embedding and masked inference queries on Ginkgo's 3'UTR language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    three_utr_mean_embedding_params,
    three_utr_masked_inference_params,
)

client = GinkgoAIClient()

# Simple query for embedding computation
prediction = client.query(three_utr_mean_embedding_params("ATTGCG"))
# prediction["embedding"] == [1.05, -2.34, ...]


# Simple query for masked inference
prediction = client.query(three_utr_masked_inference_params("ATT<mask>TAC"))

queries = [
    three_utr_mean_embedding_params("AGCGC"),
    three_utr_mean_embedding_params("ATTGCG"),
    three_utr_mean_embedding_params("TACCGCA"),
]
predictions = client.batch_query(queries)

# predictions[0]["result"]["embedding"] == [1.05, -2.34, ...]
