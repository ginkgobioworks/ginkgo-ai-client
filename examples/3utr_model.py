"""In this example we compute embedding and run masked inference
 on Ginkgo's 3'UTR language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
)

client = GinkgoAIClient()
model = "ginkgo-maskedlm-3utr-v1"

# SIMPLE QUERY FOR EMBEDDING COMPUTATION

query = MeanEmbeddingQuery(sequence="ATTGCG", model=model)
prediction = client.send_request(query)
# prediction.embedding == [1.05, -2.34, ...]


# SIMPLE QUERY FOR MASKED INFERENCE

query = MaskedInferenceQuery(sequence="ATT<mask>TAC", model=model)
prediction = client.send_request(query)

# BATCH REQUEST

queries = [
    MeanEmbeddingQuery(sequence=sequence, model=model)
    for sequence in ["AGCGC", "ATTGCG", "TACCGCA"]
]
predictions = client.send_batch_request(queries)
# predictions[0].embedding == [1.05, -2.34, ...]
