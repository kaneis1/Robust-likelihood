# Rerun with the typicality definition

This adds the mathematical definition to the actual question sent to Jev, GPT and Claude.
The previous likely and typical prompts remain unchanged. This is a new measurement:
confidence that an utterance falls within a specified typicality band, not P(E|H).

The definition is S=-ln p(U|H), h=E[-ln p(V|H)], D=|S-h|; membership means D<=epsilon.
The unit is one complete utterance, with its length and stopping convention. This is
the M=1 specialization of the batch statistic discussed by Nalisnick et al. (2019),
arXiv:1906.02994v2, not a guarantee of typical-set concentration for individual strings.

## Required specification

Copy configs/formal_typicality.template.json to configs/formal_typicality.local.json.
Fill reference_distribution with the sampling process conditional on the stated H,
including utterance length, stopping and context. Set a justified positive epsilon_nats.
The template deliberately has no invented distribution or threshold. A verbal description
of natural user behavior yields subjective estimates, not analytically verified targets.
A fully specified finite generator permits exact membership labels. Do not score existing
alarm intent labels as formal typicality labels. Do not assume empty or short strings
must be atypical without specifying the generator.

## Run in Cursor, from the repository with the existing local API credentials

    python -m robust_likelihood.formal_rerun --spec configs/formal_typicality.local.json --dataset synthetic_v2 --models jev,gpt,claude --dry-run

After inspecting the plan, repeat the command without --dry-run for live calls.
For the family experiment use --dataset synthetic_families --split holdout.
Use the dataset and split of the previous run for a paired comparison. The new command
does not infer which uncommitted run was last executed. --max-requests limits HTTP calls,
including retries; it is not a balanced subsample. --repeats preserves explicit repeat IDs.

The prompt content determines a unique version and cache identity. Each manifest stores
the full prompt, reference and threshold. Raw responses, errors, model IDs, and usage
use the existing runner. Classification is excluded because it is a different task.
Automatic intent-accuracy evaluation is disabled for formal runs; metrics.json marks
raw_scores_only. Compare shared (example, hypothesis, provider, repeat) pairs across runs.
Do not run the legacy intent evaluator on these outputs. Live reruns require local keys.

## Status on 12 October 2026

Prepared and tested without model APIs. No provider credentials were available in the
authoring environment. No live results are claimed and no previous report is overwritten.
The reference distribution, threshold, and exact prior local run remain to be supplied.
