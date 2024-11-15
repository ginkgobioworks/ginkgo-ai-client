# Ginkgo's AI model API client

**Work in progress: this repo was just made public and we are still working on integration**

A python client for [Ginkgo's AI model API](https://models.ginkgobioworks.ai/), to run inference on public and Ginkgo-proprietary models.
Learn more in the [Model API announcement](https://www.ginkgobioworks.com/2024/09/17/ginkgo-model-api-ai-research/).

## Prerequisites

Register at https://models.ginkgobioworks.ai/ to get credits and an API KEY (of the form `xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx`).
Store the API KEY in the `GINKGOAI_API_KEY` environment variable.

## Installation

Install the python client with pip:

```bash
pip install ginkgo-ai-client
```

## Usage:

**Note: This is an alpha version of the client and its interface may vary in the future.**

**Example : masked inference with Ginkgo's AA0 model**

The client requires an API key (and defaults to `os.environ.get("GINKGOAI_API_KEY")` if none is explicitly provided)

```python
from ginkgo_ai_client import GinkgoAIClient, MaskedInferenceQuery

client = GinkgoAIClient()
model = "ginkgo-aa0-650M"

# SINGLE QUERY

query = MaskedInferenceQuery(sequence="MPK<mask><mask>RRL", model=model)
prediction = client.send_request(query)
# prediction.sequence == "MPKRRRRL"

# BATCH QUERY

sequences = ["MPK<mask><mask>RRL", "M<mask>RL", "MLLM<mask><mask>R"]
queries = [MaskedInferenceQuery(sequence=seq, model=model) for seq in sequences]
predictions = client.send_batch_request(queries)
# predictions[0].sequence == "MPKRRRRL"
```

Changing the `model` parameter to `esm2-650M` or `esm2-3b` in this example will perform
masked inference with the ESM2 model.

**Example : embedding computation with Ginkgo's 3'UTR language model**

```python
from ginkgo_ai_client import GinkgoAIClient, MeanEmbeddingQuery

client = GinkgoAIClient()
model = "ginkgo-maskedlm-3utr-v1"

# SINGLE QUERY

query = MeanEmbeddingQuery(sequence="ATTGCG", model=model)
prediction = client.send_request(query)
# prediction.embedding == [1.05, -2.34, ...]

# BATCH QUERY

sequences = ["ATTGCG", "CAATGC", "GCGCACATGT"]
queries = [MeanEmbeddingQuery(sequence=seq, model=model) for seq in sequences]
predictions = client.send_batch_request(queries)
# predictions[0].embedding == [1.05, -2.34, ...]
```

## Available models

See the [example folder](examples/) and [reference docs](https://ginkgobioworks.github.io/ginkgo-ai-client/) for more details on usage and parameters.

| Model | Description                                 | Reference                                                                                    | Supported queries            | Versions |
| ----- | ------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------- | -------- |
| ESM2  | Large Protein language model from Meta      | [Github](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold)                 | Embeddings, masked inference | 3B, 650M |
| AA0   | Ginkgo's proprietary protein language model | [Announcement](https://www.ginkgobioworks.com/2024/09/17/aa-0-protein-llm-technical-review/) | Embeddings, masked inference | 650M     |
| 3UTR  | Ginkgo's proprietary 3'UTR language model   | [Preprint](https://www.biorxiv.org/content/10.1101/2024.10.07.616676v1)                      | Embeddings, masked inference | v1       |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Releases

Make sure the changelog is up to date nd the top section reads `Unreleased`, increment the version with `bumpversion` and upload with tags:

```
bumpversion patch|minor|major
git push && git push --tags
```

This should create a release on Github and publish to PyPI.
