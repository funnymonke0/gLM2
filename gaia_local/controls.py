import math
import random

from gaia_local.constants import AMINO_ACIDS
from gaia_local.types import SequenceRecord


def sequence_identity(query: str, candidate: str) -> float:
    max_len = max(len(query), len(candidate))
    if max_len == 0:
        return 1.0
    matches = sum(left == right for left, right in zip(query, candidate))
    return matches / max_len


def mutate_sequence(sequence: str, mutation_rate: float = 0.03, seed: int = 13) -> str:
    rng = random.Random(seed)
    residues = list(sequence)
    mutation_count = max(1, math.ceil(len(residues) * mutation_rate))
    indices = rng.sample(range(len(residues)), k=mutation_count)
    for index in indices:
        original = residues[index]
        choices = [aa for aa in AMINO_ACIDS if aa != original]
        residues[index] = rng.choice(choices)
    return "".join(residues)


def make_control_records(query: SequenceRecord) -> list[SequenceRecord]:
    shuffled = list(query.sequence)
    random.Random(17).shuffle(shuffled)
    return [
        SequenceRecord("control_exact", "control_exact", query.sequence, query.source_path),
        SequenceRecord("control_mutant", "control_mutant", mutate_sequence(query.sequence), query.source_path),
        SequenceRecord("control_shuffled", "control_shuffled", "".join(shuffled), query.source_path),
        SequenceRecord("control_reversed", "control_reversed", query.sequence[::-1], query.source_path),
    ]