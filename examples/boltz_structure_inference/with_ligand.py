"""Simple example where we predict the 3D structure of the GFP protein."""

from ginkgo_ai_client import GinkgoAIClient, BoltzStructurePredictionQuery

client = GinkgoAIClient()
query = BoltzStructurePredictionQuery.from_yaml_file("with_ligand.yaml")
print("Sending the request, it might take a hot minute...")
response = client.send_request(query, timeout=1000)
response.download_structure("with_ligand.cif")
