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
