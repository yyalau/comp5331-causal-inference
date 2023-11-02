from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, TypeVar, Tuple
import random

U = TypeVar('U')
V = TypeVar('V')


def get_flattened_index(
    dictionary: Mapping[U, List[V]],
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


def sample_sequence_delete(
    sequence: List[V]
) -> V:

    """
    Sample one element from a sequence and delete it from that sequence.
    """
    sample_index = random.choice(list(range(len(sequence))))
    return sequence.pop(sample_index)

def sample_sequence_no_replace(
    value: List[U],
    num_samples: int,
    predicate: Optional[Callable[[U], bool]] = None
) -> List[U]:
    """
    Samples a specified number of elements from a given list without
    replacement, optionally filtering them based on a provided predicate

    Raises:
        ValueError: If the number of samples requested is larger
        than the dictionary size.

    """
    random.shuffle(value)
    samples = list(filter(predicate, value))

    if num_samples > len(samples):
        raise ValueError(
            "Number of samples requested is larger than the dictionary size."
        )
    return samples[:num_samples]


def sample_sequence_replace(
    value: List[U],
    num_samples: int,
    predicate: Optional[Callable[[U], bool]] = None
) -> List[U]:
    """
    Samples a specified number of elements from a given list with
    replacement, optionally filtering them based on a provided predicate.
    """
    samples = random.sample(value, num_samples)
    return list(filter(predicate, samples))


def sample_dictionary(
    dictionary: Mapping[U, Any],
    num_samples: int,
    replace: bool,
    predicate: Optional[Callable[[U], bool]] = None
) -> List[U]:
    """
    Samples a specified number of keys from a given dictionary,
    with the option to sample with or without replacement,
    and optionally apply a predicate to filter the sampled keys.
    """
    keys = list(dictionary.keys())
    if replace:
        return sample_sequence_replace(keys, num_samples, predicate)
    return sample_sequence_no_replace(keys, num_samples, predicate)
