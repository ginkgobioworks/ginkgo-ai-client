"""In this example for generating 3' and 5' UTRs as well
as codon sequence for a given protein sequence of interest for
encoding in a linear mRNA."""


from ginkgo_ai_client.queries import RNADiffusionMaskedQuery
from ginkgo_ai_client import (
    GinkgoAIClient,
)

client = GinkgoAIClient()
model = "mrna-foundation"

# SIMPLE QUERY FOR GENERATING PARTIAL/FULLY MASKED UTRs

client = GinkgoAIClient()
three_utr="<mask>" * 20
five_utr="AAA<mask>TTTGGGCC<mask><mask>"
protein_sequence="MAKS-" # '-' denotes end of protein sequence
species="HOMO_SAPIENS"

query = RNADiffusionMaskedQuery(
    three_utr=three_utr,
    five_utr=five_utr,
    protein_sequence=protein_sequence,
    species=species,
    model=model,
    temperature=1.0,
    decoding_order_strategy="entropy",
    unmaskings_per_step=10,
    num_samples=1
)
response = client.send_request(query)
samples = response.samples

