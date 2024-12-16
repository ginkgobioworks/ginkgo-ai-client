import tempfile
from pathlib import Path
import pytest

from ginkgo_ai_client import (
    GinkgoAIClient,
    MaskedInferenceQuery,
    MeanEmbeddingQuery,
    PromoterActivityQuery,
    DiffusionMaskedQuery,
    BoltzStructurePredictionQuery,
)


@pytest.mark.parametrize(
    "model, sequence, expected_sequence",
    [
        ("ginkgo-aa0-650M", "MCL<mask>YAFVATDA<mask>DDT", "MCLLYAFVATDADDDT"),
        ("esm2-650M", "MCL<mask>YAFVATDA<mask>DDT", "MCLLYAFVATDAADDT"),
        ("ginkgo-maskedlm-3utr-v1", "ATTG<mask>G", "ATTGGG"),
        ("lcdna", "ATRGAyAtg<mask>TAC<mask>", "atggatatgtta<unk>"),
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
        source="expression",
        tissue_of_interest={
            "heart": ["CNhs10608+", "CNhs10612+"],
            "liver": ["CNhs10608+", "CNhs10612+"],
        },
    )

    response = client.send_request(query)
    assert "heart" in response.activity_by_tissue
    assert "liver" in response.activity_by_tissue


@pytest.mark.parametrize(
    "model, sequence",
    [
        ("lcdna", "ATRGAyAtg<mask>TAC<mask>"),
        ("abdiffusion", "MCL<mask>YAFVATDA<mask>DDT"),
    ],
)
def test_diffusion_masked_inference(model, sequence):
    client = GinkgoAIClient()
    query = DiffusionMaskedQuery(
        sequence=sequence,  # upper and lower cases
        model=model,
        temperature=0.5,
        decoding_order_strategy="entropy",
        unmaskings_per_step=2,
    )
    response = client.send_request(query)
    assert isinstance(response.sequence, str)
    assert "<mask>" not in response.sequence


def test_boltz_structure_prediction():
    client = GinkgoAIClient()
    data_file = Path(__file__).parent / "data" / "boltz_input_single_chain.yaml"
    query = BoltzStructurePredictionQuery.from_yaml_file(data_file)
    response = client.send_request(query)
    with tempfile.TemporaryDirectory() as temp_dir:
        response.download_structure(Path(temp_dir) / "structure.cif")
        response.download_structure(Path(temp_dir) / "structure.pdb")
