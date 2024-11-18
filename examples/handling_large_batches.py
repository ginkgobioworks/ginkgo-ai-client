"""In this example, we open a large fasta file with thousands of sequence
(could be millions) and send it little batch by little batch to the API, and save
the outputs (here embeddings) to disk."""

from pathlib import Path
import json

from ginkgo_ai_client import GinkgoAIClient, MeanEmbeddingQuery

input_file = Path(__file__).parent / "data" / "100_dna_sequences.fasta"
output_folder = Path(__file__).parent / "outputs" / "large_batches"
output_folder.mkdir(parents=True, exist_ok=True)

client = GinkgoAIClient()
model = "ginkgo-maskedlm-3utr-v1"
queries = MeanEmbeddingQuery.iter_from_fasta(input_file, model=model)
for batch_result in client.send_requests_by_batches(queries, batch_size=10):
    for query_result in batch_result:
        with open(output_folder / f"{query_result.query_name}.json", "w") as f:
            json.dump(query_result.dict(), f)
