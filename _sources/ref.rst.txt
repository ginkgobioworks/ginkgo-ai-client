API Documentation
-----------------

GinkgoAIClient
~~~~~~~~~~~~~~

.. automodule:: ginkgo_ai_client.client
   :members:

Mean embedding Queries
~~~~~~~~~~~~~~~~~~~~~~

Used to get embedding vectors for protein or nucleotide sequences, using models such as ESM, Ginkgo-AA0, etc.

.. autoclass:: ginkgo_ai_client.queries.MeanEmbeddingQuery
.. autoclass:: ginkgo_ai_client.queries.EmbeddingResponse

Masked inference queries
~~~~~~~~~~~~~~~~~~~~~~~~

Used to get maximum-likelihood predictions for masked  protein or nucleotide sequences, using models such as ESM, Ginkgo-AA0, etc.

.. autoclass:: ginkgo_ai_client.queries.MaskedInferenceQuery
.. autoclass:: ginkgo_ai_client.queries.SequenceResponse


Promoter activity prediction queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to predict the activity of promoters in various human tissues, using Borzoi and Ginkgo's Promoter-0

.. autoclass:: ginkgo_ai_client.queries.PromoterActivityQuery
.. autoclass:: ginkgo_ai_client.queries.PromoterActivityResponse

Diffusion queries
~~~~~~~~~~~~~~~~~

Used to generate protein or nucleotide sequences using Ginkgo-devloped diffusion models LCDNA and AB-Diffusion.

.. autoclass:: ginkgo_ai_client.queries.DiffusionMaskedQuery
.. autoclass:: ginkgo_ai_client.queries.DiffusionMaskedResponse


mRNA Diffusion queries
~~~~~~~~~~~~~~~~~~~~~~

Used to linear mRNA sequences using Ginkgo-devloped diffusion models mRNA.

.. autoclass:: ginkgo_ai_client.queries.RNADiffusionMaskedQuery
.. autoclass:: ginkgo_ai_client.queries.MultimodalDiffusionMaskedResponse

Boltz structure inference queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to predict the 3D structure of a protein sequence using Boltz.

.. autoclass:: ginkgo_ai_client.queries.BoltzStructurePredictionQuery
.. autoclass:: ginkgo_ai_client.queries.BoltzStructurePredictionResponse