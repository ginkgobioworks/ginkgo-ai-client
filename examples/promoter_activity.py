"""
This example shows how to use the PromoterActivityQuery to predict the activity of a
promoter in different tissues, based on the Borzoi model.
"""

from pathlib import Path
from ginkgo_ai_client import GinkgoAIClient, PromoterActivityQuery

client = GinkgoAIClient()
orf_sequence = "tgccagccatctgttgtttgcc"
promoter_sequence = "GTCCCACTGATGAACTGTGCT"


query = PromoterActivityQuery(
    promoter_sequence=promoter_sequence,
    orf_sequence=orf_sequence,
    source="expression",
    tissue_of_interest={
        "heart": ["CNhs10608+", "CNhs10612+"],
        "liver": ["CNhs10608+", "CNhs10612+"],
    },
)

response = client.send_request(query)
print("Single-query response:", response)


# In this next example we pull the promoter files from a fasta file and send them
# in batches, writing the results to a JSONL, as they arrive.

fasta_path = Path(__file__).parent / "data" / "100_dna_sequences.fasta"
queries = PromoterActivityQuery.iter_with_promoter_from_fasta(
    fasta_path=fasta_path,
    orf_sequence=orf_sequence,
    source="expression",
    tissue_of_interest={
        "heart": ["CNhs10608+", "CNhs10612+"],
        "liver": ["CNhs10608+", "CNhs10612+"],
    },
)

print("Now sending 100 requests, by batches of 10")
print("Writing results to promoter_activity.jsonl...")
output_file = Path(__file__).parent / "outputs" / "promoter_activity.jsonl"
for batch_result in client.send_requests_by_batches(queries, batch_size=10):
    for query_result in batch_result:
        query_result.write_to_jsonl(output_file)
