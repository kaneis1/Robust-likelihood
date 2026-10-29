"""Hypotheses, roll sequences, exact likelihoods, and the three scorer prompts [Frozen].

Each (k, generating die) cell draws its sequences from its own random.Random, seeded
with int(sha256(f"{seed}:dice:{generator}:k{k}").hexdigest()[:16], 16), so a cell can
be regenerated alone. Every sequence is scored under all three dice. Normalization
queries add every face at k = 1 under each die and all 36 sequences at k = 2 under
the fair die.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from robust_likelihood.multiwoz.reference import bootstrap_rng

FACES = (1, 2, 3, 4, 5, 6)

S1_SYSTEM = "You simulate dice rolls exactly as described."
S1_USER = (
    "The die: {wording}. Roll it {k} times and report the results in order as a comma-separated list of numbers. "
    "Reply with the list only."
)
S2_USER = (
    "Someone rolled {wording} {k} times and reported the results in order: {sequence}. "
    "How likely is it that they would report exactly this sequence? "
    "Answer with a single probability between 0 and 1 and nothing else."
)
S3_USER = (
    "{Wording} is rolled {k} times. What is the probability that the results, in order, are exactly {sequence}? "
    "Answer with a single number between 0 and 1 and nothing else."
)
FRAMINGS = {"S2": "judgment", "S3": "computation"}


def face_probabilities(config: dict) -> dict[str, dict[int, Fraction]]:
    found = {}
    for name, spec in config["hypotheses"].items():
        probabilities = {int(face): Fraction(value) for face, value in spec["faces"].items()}
        if sorted(probabilities) != list(FACES) or sum(probabilities.values()) != 1:
            raise ValueError(f"die {name} must give the six faces probabilities that sum to 1")
        found[name] = probabilities
    return found


def sequence_text(faces) -> str:
    return ", ".join(str(face) for face in faces)


def log_likelihood(faces, probabilities: dict[int, Fraction]) -> float | None:
    """Natural log of the exact probability of the ordered sequence; None when it is impossible."""
    if any(probabilities[face] == 0 for face in faces):
        return None
    return sum(math.log(probabilities[face]) for face in faces)


def draw_sequences(config: dict) -> list[dict]:
    dice = face_probabilities(config)
    count = int(config["sequences_per_cell"])
    api_count = int(config["api_sequences_per_cell"])
    rows = []
    for k in config["rolls"]:
        for generator, probabilities in dice.items():
            rng = bootstrap_rng(int(config["seed"]), f"dice:{generator}:k{k}")
            weights = [float(probabilities[face]) for face in FACES]
            for index in range(count):
                faces = rng.choices(FACES, weights=weights, k=int(k))
                rows.append({
                    "seq_id": f"k{int(k):02d}-{generator}-{index:02d}",
                    "k": int(k),
                    "generator": generator,
                    "index": index,
                    "faces": faces,
                    "text": sequence_text(faces),
                    "api_subset": index < api_count,
                })
    return rows


def build_pairs(config: dict, sequences: list[dict]) -> list[dict]:
    dice = face_probabilities(config)
    rows = []
    for sequence in sequences:
        for hypothesis, probabilities in dice.items():
            logp = log_likelihood(sequence["faces"], probabilities)
            rows.append({
                "query_id": f"{sequence['seq_id']}|{hypothesis}",
                "kind": "pair",
                "seq_id": sequence["seq_id"],
                "k": sequence["k"],
                "generator": sequence["generator"],
                "hypothesis": hypothesis,
                "faces": sequence["faces"],
                "text": sequence["text"],
                "own": sequence["generator"] == hypothesis,
                "api_subset": sequence["api_subset"],
                "true_logp": logp,
                "true_log10": None if logp is None else logp / math.log(10),
            })
    return rows


def build_normalization(config: dict) -> list[dict]:
    dice = face_probabilities(config)
    sets = [("k1", hypothesis, [(face,) for face in FACES]) for hypothesis in dice]
    sets.append(("k2", "fair", list(itertools.product(FACES, repeat=2))))
    rows = []
    for set_name, hypothesis, outcomes in sets:
        for faces in outcomes:
            logp = log_likelihood(faces, dice[hypothesis])
            rows.append({
                "query_id": f"norm|{set_name}|{hypothesis}|{''.join(str(face) for face in faces)}",
                "kind": "norm",
                "norm_set": f"{set_name}_{hypothesis}",
                "k": len(faces),
                "hypothesis": hypothesis,
                "faces": list(faces),
                "text": sequence_text(faces),
                "api_subset": True,
                "true_p": float(math.prod(dice[hypothesis][face] for face in faces)),
                "true_logp": logp,
            })
    return rows


def s1_messages(wording: str, k: int) -> list[dict]:
    return [{"role": "system", "content": S1_SYSTEM}, {"role": "user", "content": S1_USER.format(wording=wording, k=k)}]


def verbal_messages(scorer: str, wording: str, k: int, sequence: str) -> list[dict]:
    if scorer == "S2":
        text = S2_USER.format(wording=wording, k=k, sequence=sequence)
    elif scorer == "S3":
        text = S3_USER.format(Wording=wording[0].upper() + wording[1:], k=k, sequence=sequence)
    else:
        raise ValueError(scorer)
    return [{"role": "user", "content": text}]


def pilot_pairs(pairs: list[dict], count: int, seed: int) -> list[dict]:
    """A seeded sample of the API subset, in query order."""
    subset = [row for row in pairs if row["api_subset"]]
    chosen = set(bootstrap_rng(seed, "dice:pilot").sample(range(len(subset)), count))
    return [row for index, row in enumerate(subset) if index in chosen]
