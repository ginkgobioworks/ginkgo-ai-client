"""This file tests the batching functionality of the client, using short DNA sequences
for which we'll compute embeddings."""

from pathlib import Path
import itertools
from ginkgo_ai_client import GinkgoAIClient, MeanEmbeddingQuery, MaskedInferenceQuery
from ginkgo_ai_client.queries import EmbeddingResponse


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


def test_that_batch_results_are_in_the_same_order_as_queries():
    """We test that the results are consistent with the queries in the same order"""
    client = GinkgoAIClient()
    prefixes = ["".join(x) for x in itertools.product("ACGT", repeat=3)]
    masked_sequences = [prefix + "<mask>" for prefix in prefixes]
    model = "ginkgo-maskedlm-3utr-v1"
    queries = [
        MaskedInferenceQuery(sequence=seq, model=model, query_name=prefix)
        for seq, prefix in zip(masked_sequences, prefixes)
    ]
    results = client.send_batch_request(queries)
    for result, prefix in zip(results, prefixes):
        assert result.query_name == prefix
        assert result.sequence.startswith(prefix)


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
