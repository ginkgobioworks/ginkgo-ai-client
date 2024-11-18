"""This file tests the batching functionality of the client, using short DNA sequences
for which we'll compute embeddings."""

from ginkgo_ai_client import GinkgoAIClient, MeanEmbeddingQuery
from ginkgo_ai_client.queries import EmbeddingResponse
from pathlib import Path

FASTA_FILE = Path(__file__).parent / "data" / "50_dna_sequences.fasta"
model = "ginkgo-maskedlm-3utr-v1"


def test_that_send_batch_request_works():
    """We test that this function returns the expected number of results, and that
    the results are not errored."""
    client = GinkgoAIClient()
    queries = MeanEmbeddingQuery.list_from_fasta(FASTA_FILE, model=model)
    results = client.send_batch_request(queries)
    assert len(results) == 50
    assert all(isinstance(r, EmbeddingResponse) for r in results)


def test_that_send_requests_by_batches_works():
    """We test that this function returns the expected number of batches, with the
    correct batch size, and that the results are not errored."""
    queries = MeanEmbeddingQuery.iter_from_fasta(FASTA_FILE, model=model)
    client = GinkgoAIClient()
    counter = 0
    for batch_result in client.send_requests_by_batches(queries, batch_size=10):
        counter += 1
        assert len(batch_result) == 10
        for query_result in batch_result:
            assert isinstance(query_result, EmbeddingResponse)
    assert counter == 5
