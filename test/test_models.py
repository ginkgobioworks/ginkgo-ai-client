import pytest

from ginkgo_ai_client import (
    GinkgoAIClient,
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
    PromoterActivityQuery,
)


@pytest.mark.parametrize(
    "model, sequence, expected_sequence",
    [
        ("ginkgo-aa0-650M", "MCL<mask>YAFVATDA<mask>DDT", "MCLLYAFVATDADDDT"),
        ("esm2-650M", "MCL<mask>YAFVATDA<mask>DDT", "MCLLYAFVATDAADDT"),
        ("ginkgo-maskedlm-3utr-v1", "ATTG<mask>G", "ATTGGG"),
    ],
)
def test_masked_inference(model, sequence, expected_sequence):
    client = GinkgoAIClient()
    results = client.send_request(MaskedInferenceQuery(sequence=sequence, model=model))
    assert results.sequence == expected_sequence


@pytest.mark.parametrize(
    "model, sequence, expected_length",
    [
        ("ginkgo-aa0-650M", "MCLYAFVATDADDT", 1280),
        ("esm2-650M", "MCLYAFVATDADDT", 1280),
        ("ginkgo-maskedlm-3utr-v1", "ATTGGG", 768),
    ],
)
def test_embedding_inference_query(model, sequence, expected_length):
    client = GinkgoAIClient()
    results = client.send_request(MeanEmbeddingQuery(sequence=sequence, model=model))
    assert len(results.embedding) == expected_length


def test_batch_AA0_masked_inference():
    client = GinkgoAIClient()
    sequences = ["M<mask>P", "M<mask>R", "M<mask>S"]
    batch = [
        MaskedInferenceQuery(sequence=s, model="ginkgo-aa0-650M") for s in sequences
    ]
    results = client.send_batch_request(batch)
    assert [r.sequence for r in results] == ["MPP", "MRR", "MSS"]


def test_promoter_activity():
    client = GinkgoAIClient()
    query = PromoterActivityQuery(
        promoter_sequence="tgccagccatctgttgtttgcc",
        orf_sequence="GTCCCACTGATGAACTGTGCT",
        tissue_of_interest={
            "heart": ["CNhs10608+", "CNhs10612+"],
            "liver": ["CNhs10608+", "CNhs10612+"],
        },
    )

    response = client.send_request(query)
    assert "heart" in response.activity_by_tissue
    assert "liver" in response.activity_by_tissue
