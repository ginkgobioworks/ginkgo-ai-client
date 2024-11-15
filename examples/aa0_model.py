"""In this example we compute embedding and run masked inference
 on the aa0 language model."""

from ginkgo_ai_client import (
    GinkgoAIClient,
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
)

client = GinkgoAIClient()
model = "ginkgo-aa0-650M"

# SIMPLE QUERY FOR EMBEDDING COMPUTATION

query = MeanEmbeddingQuery(sequence="MLYLRRL", model=model)
prediction = client.send_request(query)
# prediction.embedding == [1.05, -2.34, ...]


# SIMPLE QUERY FOR MASKED INFERENCE

query = MaskedInferenceQuery(sequence="MLY<mask>RRL", model=model)
prediction = client.send_request(query)
# prediction.sequence == "MLYRRL"

# BATCH REQUEST

queries = [
    MeanEmbeddingQuery(sequence=sequence, model=model)
    for sequence in ["MLYLRRL", "MLL", "MLYLLRRL"]
]
predictions = client.send_batch_request(queries)
# predictions[0].embedding == [1.05, -2.34, ...]
