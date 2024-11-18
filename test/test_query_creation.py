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
        tissue_of_interest={
            "heart": ["CNhs10608+", "CNhs10612+"],
            "liver": ["CNhs10608+", "CNhs10612+"],
        },
    )
    assert len(queries) == 50
