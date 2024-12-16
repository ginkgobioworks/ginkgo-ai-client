"""Simple example where we predict the 3D structure of the GFP protein."""

from ginkgo_ai_client import GinkgoAIClient, BoltzStructurePredictionQuery

client = GinkgoAIClient()
sequence = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTL"
    "VTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLV"
    "NRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLAD"
    "HYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

query = BoltzStructurePredictionQuery.from_protein_sequence(sequence)
response = client.send_request(query)
response.download_structure("GFP.pdb")
