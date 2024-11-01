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
from ginkgo_ai_client import GinkgoAIClient, aa0_masked_inference_params

client = GinkgoAIClient()
prediction = client.query(aa0_masked_inference_params("MPK<mask><mask>RRL"))
# prediction["sequence"] == "MPKYLRRL"

predictions = client.batch_query([
    aa0_masked_inference_params("MPK<mask><mask>RRL"),
    aa0_masked_inference_params("M<mask>RL"),
    aa0_masked_inference_params("MLLM<mask><mask>R"),
])
# predictions[0]["result"]["sequence"] == "MPKYLRRL"
```

Note that you can get esm predictions by using `esm_masked_inference_params` in the example above.

**Example : embedding computation with Ginkgo's 3'UTR language model**

```python
from ginkgo_ai_client import GinkgoAIClient, three_utr_mean_embedding_params

client = GinkgoAIClient()
prediction = client.query(three_utr_mean_embedding_params("ATTGCG"))
# prediction["embedding"] == [1.05, -2.34, ...]

predictions = client.batch_query([
    three_utr_mean_embedding_params("ATTGCG"),
    three_utr_mean_embedding_params("CAATGC"),
    three_utr_mean_embedding_params("GCGCACATGT"),
])
# predictions[0]["result"]["embedding"] == [1.05, -2.34, ...]
```

## Available models

See the [example folder](examples/) and [reference docs](https://github.com/ginkgobioworks/ginkgo-ai-client/index.html) for more details on usage and parameters.

| Model | Description                                 | Reference                                                                                    | Supported queries            | Versions |
| ----- | ------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------- | -------- |
| ESM2  | Large Protein language model from Meta      | [Github](https://github.com/facebookresearch/esm?tab=readme-ov-file#esmfold)                 | Embeddings, masked inference | 3B, 650M |
| AA0   | Ginkgo's proprietary protein language model | [Announcement](https://www.ginkgobioworks.com/2024/09/17/aa-0-protein-llm-technical-review/) | Embeddings, masked inference | 650M     |
| 3UTR  | Ginkgo's proprietary 3'UTR language model   | [Preprint](https://www.biorxiv.org/content/10.1101/2024.10.07.616676v1)                      | Embeddings, masked inference | 650M     |

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
