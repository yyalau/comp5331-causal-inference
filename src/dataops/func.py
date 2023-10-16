from collections.abc import MutableSequence, Sequence
from typing import Any, Callable, Mapping, Optional, TypeVar, Tuple
import random

U = TypeVar('U')
V = TypeVar('V')


def get_flattened_index(
    dictionary: Mapping[U, Sequence[V]],
    index: int
) -> Tuple[U, V]:
    """
    Get the element at the specified flattened index from a dictionary of
    sequences.

    Raises:
        StopIteration: If the index is out of bounds.

    """
    total_length = 0
    for key, seq in dictionary.items():
        if total_length + len(seq) > index:
            return key, seq[index - total_length]
        total_length += len(seq)

    raise StopIteration("Index out of bound")


def sample_sequence_and_remove_from_population(
    sequence: MutableSequence[V],
    num_samples: int
) -> Sequence[V]:

    """
    Sample elements from a sequence and deletes the elements from that
    sequence.

    Randomly selects a specified number of elements from
    the provided mutable sequence and removes them from the sequence.

    """
    sample_indices = random.sample(list(range(len(sequence))), num_samples)
    samples = [sequence[i] for i in sample_indices]
    for idx in sorted(sample_indices, reverse=True):
        sequence.pop(idx)

    return samples

def sample_dictionary(
    dictionary: Mapping[U, Any],
    num_samples: int,
    predicate: Optional[Callable[[U], bool]]
) -> Sequence[U]:
    """
    Sample keys from a dictionary based on predicate.

    Selects a specified number of keys from the provided
    dictionary based on the given predicate function and
    returns them as a list.

    Raises:
        ValueError: If the number of samples requested is larger
        than the dictionary size.

    """
    keys = list(dictionary.keys())
    if num_samples > len(keys):
        raise ValueError(
            "Number of samples requested is larger than the dictionary size."
        )

    random.shuffle(keys)
    if predicate is None:
        return keys[:num_samples]

    samples = [element for element in keys if predicate(element)]

    if len(samples) < num_samples:
        raise ValueError(
            "Number of samples requested is larger than the dictionary size."
        )
    return samples[:num_samples]
