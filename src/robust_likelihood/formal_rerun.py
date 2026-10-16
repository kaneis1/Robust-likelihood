"""Rerun alarm pairs with an explicit typical-set definition; no implicit reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from robust_likelihood.cli import parse_models
from robust_likelihood.prompting import PromptPack, load_prompts
from robust_likelihood.runner import run_comparison
from robust_likelihood.storage import repo_root


def make_prompt_pack(root: Path, specification: dict) -> PromptPack:
    reference = specification.get("reference_distribution")
    epsilon = specification.get("epsilon_nats")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Set reference_distribution: how complete utterances are sampled given H.")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Set epsilon_nats to a finite positive threshold; do not reuse the 0.5 score cutoff.")
    if specification.get("unit") != "complete_utterance":
        raise ValueError("This protocol requires unit=complete_utterance (not per-token probability).")
    question = (
        "Stated intention: <<INTENTION>>. "
        "Intention description: <<INTENTION_DESCRIPTION>>. "
        "Assume this stated intention H, including that full description. "
        f"Reference distribution p(U|H): {reference.strip()}\n"
        "Use the following mathematical definition of typicality, not familiarity, grammaticality, "
        "semantic compatibility, or simply high probability. U is one complete utterance, including "
        "its length and stopping point. Use natural logarithms. Its surprisal is S(U,H)=-ln p(U|H). "
        "The reference entropy is h(H)=E[V~p(.|H)][-ln p(V|H)]. "
        "For this single-utterance test, D(U,H)=|S(U,H)-h(H)|. "
        "Both much higher and much lower surprisal than this average can be atypical. "
        "This is the single-observation version; it is not a per-token average or a batch test. "
        f"Call U within the typicality band exactly when D(U,H)<={float(epsilon)} nats. "
        "If p(U|H)=0 and the reference entropy is finite, U is outside the band. "
        "Is the following utterance within that band? Your yes probability represents uncertainty "
        "about this membership claim, not the emission probability p(U|H), the distance D, "
        "or the probability of the intention. Do not replace missing distributional information "
        "with certainty about semantic fit.\nUtterance: \"<<UTTERANCE>>\""
    )
    baseline = load_prompts(root, "typical")
    # Hash all prompt text: changing reference, threshold or wrapper cannot reuse old scores.
    payload = question + "\n" + baseline.chat_pairwise
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return PromptPack(
        version=f"formal_typical_v1-{digest}", pairwise_question=question,
        classification_question=baseline.classification_question,
        chat_pairwise=baseline.chat_pairwise, chat_classification=baseline.chat_classification,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--dataset", choices=["synthetic_v2", "synthetic_families"], default="synthetic_v2")
    parser.add_argument("--split", choices=["dev", "holdout", "all"])
    parser.add_argument("--models", default="jev,gpt,claude")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    root = repo_root()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    pack = make_prompt_pack(root, spec)
    return run_comparison(
        root, providers=parse_models(args.models), dataset=args.dataset, split=args.split,
        prompt_pack=pack, prompt_version=pack.version, dry_run=args.dry_run,
        max_requests=args.max_requests, repeats=args.repeats, resume=args.resume,
    )


if __name__ == "__main__":
    main()
