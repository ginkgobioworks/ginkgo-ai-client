from ginkgo_ai_client import (
    GinkgoAIClient,
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
)


### AA0 Model


def test_AA0_masked_inference():
    client = GinkgoAIClient()
    results = client.send_request(
        MaskedInferenceQuery(
            sequence="MCL<mask>YAFVATDA<mask>DDT", model="ginkgo-aa0-650M"
        )
    )
    assert results.sequence == "MCLLYAFVATDADDDT"


def test_AA0_embedding_inference():
    client = GinkgoAIClient()
    results = client.send_request(
        MeanEmbeddingQuery(sequence="MCLYAFVATDADDT", model="ginkgo-aa0-650M")
    )
    assert len(results.embedding) == 1280


def test_batch_AA0_masked_inference():
    client = GinkgoAIClient()
    sequences = ["M<mask>P", "M<mask>R", "M<mask>S"]
    batch = [
        MaskedInferenceQuery(sequence=s, model="ginkgo-aa0-650M") for s in sequences
    ]
    results = client.send_batch_request(batch)
    assert [r.sequence for r in results] == ["MPP", "MRR", "MSS"]


### ESM Model
def test_esm_masked_inference():
    client = GinkgoAIClient()
    results = client.send_request(
        MaskedInferenceQuery(sequence="MCL<mask>YAFVATDA<mask>DDT", model="esm2-650M")
    )
    assert results.sequence == "MCLLYAFVATDAADDT"


def test_esm_embedding_inference():
    client = GinkgoAIClient()
    results = client.send_request(
        MeanEmbeddingQuery(sequence="MCLYAFVATDADDT", model="esm2-650M")
    )
    assert len(results.embedding) == 1280


# UTR model
def test_utr_masked_inference():
    client = GinkgoAIClient()
    query = MaskedInferenceQuery(
        sequence="ATTG<mask>G", model="ginkgo-maskedlm-3utr-v1"
    )
    results = client.send_request(query)
    assert results.sequence == "ATTGGG"


def test_utr_embedding_inference():
    client = GinkgoAIClient()
    query = MeanEmbeddingQuery(sequence="ATTGGG", model="ginkgo-maskedlm-3utr-v1")
    results = client.send_request(query)
    assert len(results.embedding) == 768
