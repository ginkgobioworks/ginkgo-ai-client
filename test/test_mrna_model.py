from ginkgo_ai_client.queries import RNADiffusionMaskedQuery
from Bio.Seq import Seq
from ginkgo_ai_client import (
    GinkgoAIClient,
)


def test_get_mrna_species():

    species = RNADiffusionMaskedQuery.get_species_dataframe()
    assert len(species) == 324


def test_mrna_diffusion():
    client = GinkgoAIClient()
    three_utr = "AAA<mask>TTTGGGCC<mask><mask>"
    five_utr = "AAA<mask>TTTGGGCC<mask><mask>"
    protein_sequence = "MAKS-"  # '-' for end of sequence
    species = "HOMO_SAPIENS"
    num_samples = 3
    query = RNADiffusionMaskedQuery(
        three_utr=three_utr,
        five_utr=five_utr,
        protein_sequence=protein_sequence,
        species=species,
        model="mrna-foundation",
        temperature=1.0,
        decoding_order_strategy="entropy",
        unmaskings_per_step=10,
        num_samples=num_samples,
    )

    response = client.send_request(query)

    samples = response.samples
    assert len(samples) == num_samples

    for sample in samples:
        assert "<mask>" not in sample["three_utr"]
        assert "<mask>" not in sample["five_utr"]

        # check codon sequence verbatim. +1 because of stop codon
        assert len(sample["codon_sequence"]) == len(protein_sequence) * 3
        assert sample["codon_sequence"].startswith("ATG")  # Start codon
        assert sample["codon_sequence"][-3:] in ["TAA", "TAG", "TGA"]  # stop codon

        # should translate
        translated = str(Seq(sample["codon_sequence"]).translate())
        print(translated, protein_sequence)
        assert translated.replace("*", "-") == protein_sequence
