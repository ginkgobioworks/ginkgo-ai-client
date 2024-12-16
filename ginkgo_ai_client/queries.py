"""Classes to define queries to the Ginkgo AI API."""

from typing import Dict, Optional, Any, List, Literal, Union
from abc import ABC, abstractmethod
from pathlib import Path
import json
import yaml
import tempfile

import pydantic
import requests

from ginkgo_ai_client.utils import (
    fasta_sequence_iterator,
    IteratorWithLength,
    cif_to_pdb,
)

## ---- Base classes --------------------------------------------------------------


class QueryBase(pydantic.BaseModel, ABC):
    """Base class for all queries. It's functions are:
    - Specify the mandatory class methods `to_request_params` and `parse_response`
    - Provide a better error message when a user forgets to use named arguments only.
      Without that tweak, the default error message from pydantic is very technical
      and confusing to new users.
    """

    def __new__(cls, *args, **kwargs):
        if args:
            raise TypeError(
                f"Invalid initialization: {cls.__name__} does not accept unnamed "
                f"arguments. Please name all inputs, for instance "
                f"`{cls.__name__}(field_name=value, other_field=value, ...)`."
            )
        return super().__new__(cls)

    @abstractmethod
    def to_request_params(self) -> Dict:
        pass

    @abstractmethod
    def parse_response(self, results: Dict) -> Any:
        pass


class ResponseBase(pydantic.BaseModel):

    def write_to_jsonl(self, path: str):
        with open(path, "a") as f:
            f.write(self.model_dump_json() + "\n")


## ---- MASKEDLM AND EMBEDDINGS ------------------------------------------------------

_maskedlm_models_properties = {
    "ginkgo-aa0-650M": "protein",
    "esm2-650M": "protein",
    "esm2-3B": "protein",
    "ginkgo-maskedlm-3utr-v1": "dna",
    "lcdna": "nucleotide",
    "abdiffusion": "protein",
}

_maskedlm_models_properties_str = "\n".join(
    f"- {model}: {sequence_type}"
    for model, sequence_type in _maskedlm_models_properties.items()
)


def _validate_model_and_sequence(model, sequence: str, allow_masks=False):
    """Raise an error if the model is unknown or the sequence isn't compatible."""
    valid_models = list(_maskedlm_models_properties.keys())
    if model not in valid_models:
        raise ValueError(f"Model '{model}' unknown. Sould be one of {valid_models}")
    sequence_type = _maskedlm_models_properties[model]
    if allow_masks:
        sequence = sequence.replace("<mask>", "")
    if sequence_type == "dna":
        if not set(sequence).issubset({"A", "T", "G", "C"}):
            raise ValueError(
                f"Model {model} requires the sequence to only contain ATGC characters"
            )
    elif sequence_type == "nucleotide":
        if not set(sequence.lower()).issubset(set("atgcrsywkmdbhvn")):
            raise ValueError(
                f"Model {model} requires the sequence to only contain valid ATGC or "
                f"IUPAC nucleotide characters"
            )
    elif sequence_type == "protein":
        if not set(sequence).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
            raise ValueError("Sequence must contain only protein characters")
    else:
        raise ValueError("Invalid sequence type")


class EmbeddingResponse(ResponseBase):
    """A response to a MeanEmbeddingQuery, with attributes `embedding` (the mean
    embedding of the model's last encoder layer) and `query_name` (the original
    query's name).
    """

    embedding: List[float]
    query_name: Optional[str] = None


class MeanEmbeddingQuery(QueryBase):
    """A query to infer mean embeddings from a DNA or protein sequence.

    Parameters
    ----------
    sequence: str
        The sequence to unmask. The sequence should be of the form "MLPP<mask>PPLM" with
        as many masks as desired.
    model: str
        The model to use for the inference.
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.

    Returns
    -------
    EmbeddingResponse
        ``client.send_request(query)`` returns an ``EmbeddingResponse`` with attributes
        ``embedding`` (the mean embedding of the model's last encoder layer) and
        ``query_name`` (the original query's name).

    Examples
    --------
    >>> query = MeanEmbeddingQuery("MLPP<mask>PPLM", model="ginkgo-aa0-650M")
    >>> client.send_request(query)
    EmbeddingResponse(embedding=[1.05, 0.002, ...])
    """

    sequence: str
    model: str
    query_name: Optional[str] = None

    def to_request_params(self) -> Dict:
        return {
            "model": self.model,
            "text": self.sequence,
            "transforms": [{"type": "EMBEDDING"}],
        }

    def parse_response(self, results: Dict) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=results["embedding"], query_name=self.query_name
        )

    @pydantic.model_validator(mode="after")
    def check_model_and_sequence_compatibility(cls, query):
        sequence, model = query.sequence, query.model
        _validate_model_and_sequence(model=model, sequence=sequence, allow_masks=False)
        return query

    @classmethod
    def iter_from_fasta(cls, fasta_path: str, model: str):
        """Return an iterator over the sequences in a fasta file. The iterator has
        a length attribute that gives the number of sequences in the fasta file."""
        fasta_iterator = fasta_sequence_iterator(fasta_path)
        query_iterator = (
            cls(sequence=str(record.seq), model=model, query_name=record.id)
            for record in fasta_iterator
        )
        return IteratorWithLength(query_iterator, len(fasta_iterator))

    @classmethod
    def list_from_fasta(cls, fasta_path: str, model: str):
        return list(cls.iter_from_fasta(fasta_path, model))


class SequenceResponse(ResponseBase):
    """A response to a MaskedInferenceQuery, with attributes `sequence` (the predicted
    sequence) and `query_name` (the original query's name).
    """

    sequence: str
    query_name: Optional[str] = None


class MaskedInferenceQuery(QueryBase):
    """A query to infer masked tokens in a DNA or protein sequence.

    Parameters
    ----------
    sequence: str
        The sequence to unmask. The sequence should be of the form "MLPP<mask>PPLM" with
        as many masks as desired.
    model: str
        The model to use for the inference (only "ginkgo-aa0-650M" is supported for now).
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.

    Returns
    --------
    SequenceResponse
        ``client.send_request(query)`` returns a ``SequenceResponse`` with attributes
        ``sequence` (the predicted sequence) and ``query_name`` (the original query's
        name).

    """

    sequence: str
    model: str
    query_name: Optional[str] = None

    def to_request_params(self) -> Dict:
        return {
            "model": self.model,
            "text": self.sequence,
            "transforms": [{"type": "FILL_MASK"}],
        }

    def parse_response(self, response: Dict) -> SequenceResponse:
        """The response has a sequence and the original query's name"""
        return SequenceResponse(
            sequence=response["sequence"], query_name=self.query_name
        )

    @pydantic.model_validator(mode="after")
    def check_model_and_sequence_compatibility(cls, query):
        sequence, model = query.sequence, query.model
        _validate_model_and_sequence(model=model, sequence=sequence, allow_masks=True)
        return query


auto_doc_str = f"""
    Supported inference models
    --------------------------

    Here are the supported models, and the sequence type they support. Sequences must
    be upper-case and not contain any mask etc. for embeddings computation.

    {_maskedlm_models_properties_str}
"""

for cls in [MeanEmbeddingQuery, MaskedInferenceQuery]:
    cls.__doc__ += auto_doc_str[:1]


## ---- PROMOTER ACTIVITY QUERIES ---------------------------------------------------


class PromoterActivityResponse(ResponseBase):
    """A response to a PromoterActivityQuery, with attributes `activity` (the predicted
    activity) and `query_name` (the original query's name).

    Attributes
    ----------
    activity_by_tissue: Dict[str, float]
        The activity of the promoter in each tissue.
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.
    """

    activity_by_tissue: Dict[str, float]
    query_name: Optional[str] = None


class PromoterActivityQuery(QueryBase):
    """A query to infer the activity of a promoter in different tissues.

    Parameters
    ----------
    promoter_sequence: str
        The promoter sequence. Only ATGCN characters are allowed.
    orf_sequence: str
        The ORF sequence. Only ATGCN characters are allowed.
    tissue_of_interest: Dict[str, List[str]]
        The tissues of interest, with the tracks representing each tissue, for instance
        `{"heart": ["CNhs10608+", "CNhs10612+"], "liver": ["CNhs10608+", "CNhs10612+"]}`.
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.
    model: str = "borzoi-human-fold0"
        The model to use for the inference (only one default model is supported for now).

    Returns
    -------
    PromoterActivityResponse
        ``client.send_request(query)`` returns a ``PromoterActivityResponse`` with
        attributes ``activity_by_tissue`` (the activity of the promoter in each tissue)
        and ``query_name`` (the original query's name).
    """

    promoter_sequence: str
    orf_sequence: str
    tissue_of_interest: Dict[str, List[str]]
    source: str 
    model: str = "borzoi-human-fold0"
    query_name: Optional[str] = None

    def to_request_params(self) -> Dict:
        # TODO: update the web API so the conversion isn't necessary
        data = {
            "prom": self.promoter_sequence,
            "orf": self.orf_sequence,
            "tissue_of_interest": self.tissue_of_interest,
            "source": self.source,
        }
        return {
            "model": self.model,
            "text": json.dumps(data),
            "transforms": [{"type": "PROMOTER_ACTIVITY"}],
        }

    def parse_response(self, results):
        return PromoterActivityResponse(
            query_name=self.query_name, activity_by_tissue=results
        )

    @pydantic.model_validator(mode="after")
    def sequences_are_valid_nucleotide_sequences(cls, query):
        """Raise an error if the sequences contain non-ATGCN characters."""
        query.promoter_sequence = query.promoter_sequence.upper()
        query.orf_sequence = query.orf_sequence.upper()
        if not set(query.promoter_sequence).issubset(set("ATGCN")):
            raise ValueError(
                f"Promoter sequence in query <{query.query_name}> contains "
                "non-ATGCN characters."
            )
        if not set(query.orf_sequence).issubset(set("ATGCN")):
            raise ValueError(
                f"ORF sequence in query <{query.query_name}> contains "
                "non-ATGCN characters."
            )
        return query

    @classmethod
    def iter_with_promoter_from_fasta(
        cls,
        fasta_path: str,
        orf_sequence: str,
        tissue_of_interest: Dict[str, List[str]],
        source: str,
        model: str = "borzoi-human-fold0",
    ):
        """Return an iterator of PromoterActivityQuery objects from the promoter
        sequences in a fasta file. The iterator has a length attribute that gives the
        number of sequences in the fasta file.

        Parameters
        ----------
        fasta_path: str
            The path to the fasta file containing the promoter sequences.
        orf_sequence: str
            The ORF sequence.
        tissue_of_interest: Dict[str, List[str]]
            The tissues of interest, with the tracks representing each tissue, e.g.
            `{"heart": ["CNhs10608+", "CNhs10612+"], "liver": ["CNhs10608+", "CNhs10612+"]}`.
        model: str = "borzoi-human-fold0"
            The model to use for the inference (only one default model is supported for now).
        """
        fasta_iterator = fasta_sequence_iterator(fasta_path)
        query_iterator = (
            cls(
                promoter_sequence=str(record.seq),
                orf_sequence=orf_sequence,
                tissue_of_interest=tissue_of_interest,
                source=source,
                model=model,
                query_name=record.id,
            )
            for record in fasta_iterator
        )
        return IteratorWithLength(query_iterator, len(fasta_iterator))

    @classmethod
    def list_with_promoter_from_fasta(
        cls,
        fasta_path: str,
        orf_sequence: str,
        tissue_of_interest: Dict[str, List[str]],
        source: str,
        model: str = "borzoi-human-fold0",
    ):
        """Return a list of PromoterActivityQuery objects from the promoter sequences
        in a fasta file.

        Parameters
        ----------
        fasta_path: str
            The path to the fasta file containing the promoter sequences.
        orf_sequence: str
            The ORF sequence.
        tissue_of_interest: Dict[str, List[str]]
            The tissues of interest, with the tracks representing each tissue, e.g.
            `{"heart": ["CNhs10608+", "CNhs10612+"], "liver": ["CNhs10608+", "CNhs10612+"]}`.
        model: str = "borzoi-human-fold0"
            The model to use for the inference (only one default model is supported for now).
        """
        iterator = cls.iter_with_promoter_from_fasta(
            fasta_path=fasta_path,
            orf_sequence=orf_sequence,
            tissue_of_interest=tissue_of_interest,
            source=source,
            model=model,
        )
        return list(iterator)


## ---- DIFFUSION QUERIES ---------------------------------------------------------


class DiffusionMaskedResponse(ResponseBase):
    """A response to a DiffusionMaskedQuery, with attributes `sequence` (the predicted
    sequence) and `query_name` (the original query's name).
    """

    sequence: str
    query_name: Optional[str] = None


class DiffusionMaskedQuery(QueryBase):
    """A query to perform masked sampling using a diffusion model.

    Parameters
    ----------
    sequence: str
        Input sequence for masked sampling. The sequence may contain "<mask>" tokens.
    temperature: float, optional (default=0.5)
        Sampling temperature, a value between 0 and 1.
    decoding_order_strategy: str, optional (default="entropy")
        Strategy for decoding order, must be either "max_prob" or "entropy".
    unmaskings_per_step: int, optional (default=50)
        Number of tokens to unmask per step, an integer between 1 and 1000.
    model: str
        The model to use for the inference.
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to handle exceptions.

    Returns
    -------
    DiffusionMaskedResponse
        ``client.send_request(query)`` returns a ``DiffusionMaskedResponse`` with attributes
        ``sequence`` (the predicted sequence) and ``query_name`` (the original query's name).

    Examples
    --------
    >>> query = DiffusionMaskedQuery(
    ...     sequence="ATTG<mask>TAC",
    ...     model="lcdna",
    ...     temperature=0.7,
    ...     decoding_order_strategy="entropy",
    ...     unmaskings_per_step=20,
    ... )
    >>> client.send_request(query)
    DiffusionMaskedResponse(sequence="ATTGCGTAC", query_name=None)
    """

    sequence: str
    temperature: float = 0.5
    decoding_order_strategy: str = "entropy"
    unmaskings_per_step: int = 50
    model: str
    query_name: Optional[str] = None

    def to_request_params(self) -> Dict:
        data = {
            "sequence": self.sequence,
            "temperature": self.temperature,
            "decoding_order_strategy": self.decoding_order_strategy,
            "unmaskings_per_step": self.unmaskings_per_step,
        }
        return {
            "model": self.model,
            "text": json.dumps(data),
            "transforms": [{"type": "DIFFUSION_GENERATE"}],
        }

    def parse_response(self, results: Dict) -> DiffusionMaskedResponse:
        return DiffusionMaskedResponse(
            sequence=results["sequence"][0],
            query_name=self.query_name,
        )

    @pydantic.model_validator(mode="after")
    def validate_query(cls, query):
        sequence, model = query.sequence, query.model
        # Validate sequence and model compatibility
        _validate_model_and_sequence(
            model=model,
            sequence=sequence,
            allow_masks=True,
        )
        # Validate temperature
        if not 0 <= query.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        # Validate decoding_order_strategy
        if query.decoding_order_strategy not in ["max_prob", "entropy"]:
            raise ValueError("decoding_order_strategy must be 'max_prob' or 'entropy'")
        # Validate unmaskings_per_step
        if not 1 <= query.unmaskings_per_step <= 1000:
            raise ValueError("unmaskings_per_step must be between 1 and 1000")
        return query


## ---- STRUCTURE PREDICTION QUERIES ------------------------------------------------


class _Protein(pydantic.BaseModel):
    id: Union[List[str], str]
    sequence: str

    @pydantic.validator("sequence")
    def validate_sequence(cls, sequence):
        sequence = sequence.upper()
        invalid_chars = [c for c in sequence if c not in "LAGVSERTIDPKQNFYMHWCXBUZO"]
        if len(invalid_chars) > 0:
            invalid_chars_str = ", ".join(sorted(set(invalid_chars)))
            raise ValueError(
                f"Sequence contains invalid characters: {invalid_chars_str}"
            )
        return sequence


class _CCD(pydantic.BaseModel):
    id: Union[List[str], str]
    ccd: str


class _Smiles(pydantic.BaseModel):
    id: Union[List[str], str]
    smiles: str


class BoltzStructurePredictionResponse(ResponseBase):
    """A response to a BoltzStructurePredictionQuery

    Attributes
    ----------
    cif_file_url: str
        The URL of the cif file.
    confidence_data: Dict[str, Any]
        The confidence data.
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.

    Examples
    --------

    .. code:: python

        response = BoltzStructurePredictionResponse(
            cif_file_url="https://example.com/structure.cif",
            confidence_data={"confidence": 0.95},
            query_name="my_query",
        )
        response.download_structure("structure.cif") # or...
        response.download_structure("structure.pdb")
    """

    cif_file_url: str
    confidence_data: Dict[str, Any]
    query_name: Optional[str] = None

    def download_structure(self, path: str):
        """Download the structure from the URL and save it to a file."""
        path = Path(path)
        if str(path).endswith(".pdb"):
            with tempfile.TemporaryDirectory() as temp_dir:
                cif_path = Path(temp_dir) / "temp.cif"
                self.download_structure(cif_path)
                cif_to_pdb(cif_path, path)
        else:
            response = requests.get(self.cif_file_url)
            with open(path, "w") as f:
                f.write(response.text)


class BoltzStructurePredictionQuery(QueryBase):
    """A query to predict the structure of a protein using the Boltz model.

    Parameters
    ----------
    sequences: List[Dict[Literal["protein", "ligand"], Union[_Protein, _CCD, _Smiles]]]
        The sequences to predict the structure for
    model: Literal["boltz"] = "boltz"
        The model to use for the inference (only Boltz(1) is supported for now).
    query_name: Optional[str] = None
        The name of the query. It will appear in the API response and can be used to
        handle exceptions.
    """

    sequences: List[Dict[Literal["protein", "ligand"], Union[_Protein, _CCD, _Smiles]]]
    model: Literal["boltz"] = "boltz"
    query_name: Optional[str] = None

    def to_request_params(self) -> Dict:
        return {
            "model": "boltz",
            "transforms": [{"type": "INFER_STRUCTURE"}],
            "text": self.model_dump(exclude=["model", "query_name"], mode="json"),
        }

    def parse_response(self, results: Dict) -> BoltzStructurePredictionResponse:
        return BoltzStructurePredictionResponse(
            cif_file_url=results["cif_file_url"],
            confidence_data=results["confidence_data"],
            query_name=self.query_name,
        )

    @classmethod
    def from_yaml_file(cls, path, query_name: Optional[str] = "auto"):
        path = Path(path)
        if query_name == "auto":
            query_name = path.name
        with open(path, "r") as f:
            data = yaml.load(f, yaml.SafeLoader)
        return cls(sequences=data["sequences"], query_name=query_name)

    @classmethod
    def from_protein_sequence(cls, sequence: str, query_name: Optional[str] = None):
        return cls(
            sequences=[{"protein": {"id": "A", "sequence": sequence}}],
            query_name=query_name,
        )
