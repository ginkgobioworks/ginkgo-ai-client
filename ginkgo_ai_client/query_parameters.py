"""Helpers for generating query parameters for the Ginkgo AI API."""

from typing import Dict


class TRANSFORMS:
    FILL_MASK = {"type": "FILL_MASK"}
    EMBEDDING = {"type": "EMBEDDING"}


def aa0_mean_embedding_params(sequence: str, model: str = "ginkgo-aa0-650M") -> Dict:
    """Generate the query parameters for a AA0 mean embedding query.

    The mean embedding refers to the mean of the token embedding in the encoder's
    last layer.

    Parameters
    ----------
    sequence: str
        The sequence for which to compute the mean embedding.

    model: str (default: "ginkgo-aa0-650M")
        The model to use for the embedding (only "ginkgo-aa0-650M" is supported for now).

    Query results
    -------------
    List[float]
        The mean embedding of the sequence.


    Examples
    --------

    >>> client.query(aa0_mean_embedding_params("MLPP<mask>PPLM"))
    >>> # {"embedding": [1.05, 0.002, ...]}
    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.EMBEDDING]}


def aa0_masked_inference_params(sequence: str, model: str = "ginkgo-aa0-650M") -> Dict:
    """Generate the query parameters for a masked inference query with Ginkgo's AA0
    protein-language model.

    The mean embedding of a protein sequence refers to the mean of the token
    embedding in the encoder's last layer.

    Parameters
    ----------
    sequence: str
        The sequence to unmask. The sequence should be of the form "MLPP<mask>PPLM" with
        as many masks as desired.

    model: str (default: "ginkgo-aa0-650M")
        The model to use for the inference (only "ginkgo-aa0-650M" is supported for now).

    Query results
    ------------
    {sequence: str}
        The predicted sequence where every masked position has been replaced by the
        "ATGC" nucleotide with the highest probability at this position.

    Examples
    --------

    >>> client.query(aa0_masked_inference_params("MLPP<mask>PPLM<mask>"))
    >>> # {"sequence": "MLPPKPPLMR"}

    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.FILL_MASK]}


def esm_mean_embedding_params(sequence: str, model: str = "esm2-650M") -> Dict:
    """Generate the query parameters for mean embedding inference with Ginkgo's AA0
    protein-language model.

    The mean embedding of a protein sequence refers to the mean of the token
    embedding in the encoder's last layer.

    Parameters
    ----------
    sequence: str
        The sequence for which to compute the mean embedding.

    model: str (default: "esm2-650M")
        The model to use for the embedding ("esm2-650M" or "esm2-3B").

    Query results
    ------------
    List[float]
        The mean embedding of the sequence.

    Examples
    --------

    >>> client.query(esm_mean_embedding_params("MLPP<mask>PPLM"))
    >>> # {"embedding": [1.05, 0.002, ...]}
    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.EMBEDDING]}


def esm_masked_inference_params(sequence: str, model: str = "esm2-650M") -> Dict:
    """Generate the query parameters for a ESM masked inference query.

    Parameters
    ----------
    sequence: str
        The sequence to unmask. The sequence should be of the form "MLPP<mask>PPLM" with
        as many masks as desired.

    model: str (default: "esm2-650M")
        The model to use for the inference ("esm2-650M" or "esm2-3B").

    Query results
    ------------
    {sequence: str}
        The predicted sequence where every masked position has been replaced by the
        "ATGC" nucleotide with the highest probability at this position.

    Examples
    --------

    >>> client.query(esm_masked_inference_params("MLPP<mask>PPLM<mask>"))
    >>> # {"sequence": "MLPPKPPLMR"}
    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.FILL_MASK]}


def three_utr_mean_embedding_params(
    sequence: str, model: str = "ginkgo-maskedlm-3utr-v1"
) -> Dict:
    """Generate the query parameters for a mean embedding query for Ginkgo's 3UTR
    language model.

    The mean embedding refers to the mean of the token embedding in the encoder's
    last layer.

    Parameters
    ----------
    sequence: str
        The sequence for which to compute the mean embedding, of the form "ATGC..."

    model: str (default: "ginkgo-maskedlm-3utr-v1")
        The model to use for the embedding (only "ginkgo-maskedlm-3utr-v1" is supported
        for now).

    Query results
    ------------
    List[float]
        The mean embedding of the sequence.

    Examples
    --------

    >>> client.query(three_utr_mean_embedding_params("MLPP<mask>PPLM<mask>"))
    >>> # {"embedding": [1.05, 0.002, ...]}
    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.EMBEDDING]}


def three_utr_masked_inference_params(
    sequence: str, model: str = "ginkgo-maskedlm-3utr-v1"
) -> Dict:
    """Generate the query parameters for a masked inference query for Ginkgo's 3UTR
    language model.

    Parameters
    ----------
    sequence: str
        The sequence to unmask. The sequence should be of the form "ATGC<mask>ATGC" with as
        many masks as desired.

    model: str (default: "ginkgo-maskedlm-3utr-v1")
        The model to use for the inference (only "ginkgo-maskedlm-3utr-v1" is supported
        for now).

    Query results
    ------------
    {sequence: str}
        The predicted sequence where every masked position has been replaced by the
        "ATGC" nucleotide with the highest probability at this position.
    """
    return {"model": model, "text": sequence, "transforms": [TRANSFORMS.FILL_MASK]}
