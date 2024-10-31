from ginkgo_ai_client import (
    GinkgoAIClient,
    aa0_masked_inference_params,
    aa0_mean_embedding_params,
    esm_masked_inference_params,
    esm_mean_embedding_params,
    three_utr_masked_inference_params,
    three_utr_mean_embedding_params,
)


### AA0 Model


def test_AA0_masked_inference():
    client = GinkgoAIClient()
    results = client.query(aa0_masked_inference_params("MCL<mask>YAFVATDA<mask>DDT"))
    assert results["sequence"] == "MCLLYAFVATDADDDT"


def test_AA0_embedding_inference():
    client = GinkgoAIClient()
    results = client.query(aa0_mean_embedding_params("MCLYAFVATDADDT"))
    assert len(results["embedding"]) == 1280


def test_batch_AA0_masked_inference():
    client = GinkgoAIClient()
    sequences = ["M<mask>P", "M<mask>R", "M<mask>S"]
    batch = [aa0_masked_inference_params(s) for s in sequences]
    results = client.batch_query(batch)
    print(results)
    assert [r["result"]["sequence"] for r in results] == ["MPP", "MRR", "MSS"]


### ESM Model


def test_esm_masked_inference():
    client = GinkgoAIClient()
    results = client.query(esm_masked_inference_params("MCL<mask>YAFVATDA<mask>DDT"))
    assert results["sequence"] == "MCLLYAFVATDAADDT"


def test_esm_embedding_inference():
    client = GinkgoAIClient()
    results = client.query(esm_mean_embedding_params("MCLYAFVATDADDT"))
    assert len(results["embedding"]) == 1280


# UTR model


def test_utr_masked_inference():
    client = GinkgoAIClient()
    results = client.query(three_utr_masked_inference_params("ATTG<mask>G"))
    assert results["sequence"] == "ATTGGG"


def test_utr_embedding_inference():
    client = GinkgoAIClient()
    results = client.query(three_utr_mean_embedding_params("ATTGGG"))
    assert len(results["embedding"]) == 768
