"""Utility functions for building queries, batches, etc."""

from Bio import SeqIO
from pathlib import Path
import gzip
from typing import Iterator, Union


class IteratorWithLength(Iterator):
    """An iterator that also has a length attribute, which will produce useful
    progress bars with % progress and estimated time of arrival.

    Examples
    --------

    .. code-block:: python

         some_iterator = (sequence.upper() for sequence in large_list)
         iterator_with_length = IteratorWithLength(some_iterator, length=len(large_list))

         # later in another function:
         for item in tqdm(iterator_with_length): # tqdm progress bar will be length-aware

    """

    def __init__(self, iterable, length):
        self.iterable = iter(iterable)
        self._length = length

    def __next__(self):
        return next(self.iterable)

    def __len__(self):
        return self._length


def _fast_fasta_sequence_count(fasta_path: Union[str, Path]):
    """Count the number of sequences in a fasta file by counting the ">" lines."""
    if str(fasta_path).endswith(".gz"):
        with gzip.open(fasta_path, "rt") as f:
            return sum(1 for line in f if line.startswith(">"))
    else:
        with open(fasta_path, "r") as f:
            return sum(1 for line in f if line.startswith(">"))


def fasta_sequence_iterator(fasta_path: str):
    """Return an iterator over the sequences in a fasta file. The iterator has
    a length attribute that gives the number of sequences in the fasta file."""
    # compute the number of sequences in the fasta file by counting ">"
    length = _fast_fasta_sequence_count(fasta_path)
    return IteratorWithLength(SeqIO.parse(fasta_path, "fasta"), length)
