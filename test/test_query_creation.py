import pytest
import re
from pathlib import Path
from ginkgo_ai_client.queries import MeanEmbeddingQuery, PromoterActivityQuery


def test_that_forgetting_to_name_arguments_raises_the_better_error_message():
    expected_error_message = re.escape(
        "Invalid initialization: MeanEmbeddingQuery does not accept unnamed arguments. "
        "Please name all inputs, for instance "
        "`MeanEmbeddingQuery(field_name=value, other_field=value, ...)`."
    )
    with pytest.raises(TypeError, match=expected_error_message):
        MeanEmbeddingQuery("MLLK<mask>P", model="ginkgo-aa0-650M")


def test_that_simple_query_creation_works():
    MeanEmbeddingQuery(sequence="MLLKLP", model="ginkgo-aa0-650M")


def test_that_uppercase_mask_is_accepted():
    MeanEmbeddingQuery(sequence="MLLK<MASK>P", model="ginkgo-aa0-650M")


def test_that_lowercase_protein_sequence_is_not_accepted():
    expected_error_message = re.escape(
        "Model ginkgo-aa0-650M requires the sequence to only contain the following "
        "characters: ACDEFGHIKLMNPQRSTVWY and the extra tokens ['<mask>', '<unk>', "
        "'<pad>', '<cls>', '<eos>'] (these can be upper-case). "
        "The following unparsable characters were found: k"
    )
    with pytest.raises(ValueError, match=expected_error_message):
        MeanEmbeddingQuery(sequence="MLLk<mask>P", model="ginkgo-aa0-650M")


def test_promoter_activity_query_validation():
    with pytest.raises(ValueError):
        _query = PromoterActivityQuery(
            promoter_sequence="tgccagccatctgttgtttgcc",
            orf_sequence="GTCCCAxCTGATGAAxCTGTGCT",
            tissue_of_interest={
                "heart": ["CNhs10608+", "CNhs10612+"],
                "liver": ["CNhs10608+", "CNhs10612+"],
            },
        )


def test_promoter_activity_iteration():
    fasta_path = Path(__file__).parent / "data" / "50_dna_sequences.fasta"
    queries = PromoterActivityQuery.list_with_promoter_from_fasta(
        fasta_path=fasta_path,
        orf_sequence="GTCCCACTGATGAACTGTGCT",
        source="expression",
        tissue_of_interest={
            "heart": ["CNhs10608+", "CNhs10612+"],
            "liver": ["CNhs10608+", "CNhs10612+"],
        },
    )
    assert len(queries) == 50


def test_get_tissue_tracks():
    df = PromoterActivityQuery.get_tissue_track_dataframe(tissue="heart", assay="DNASE")
    assert len(df) == 22
