# Robust likelihood pilot

Which models produce useful hypothesis–evidence scores that stay stable when wording changes and meaning does not?

These outputs are elicited plausibility judgments, not verified P(e|h).

The first comparison asks Jev and GPT the same yes/no question about each utterance:

> Assume the user has the stated intention. Is this utterance a plausible expression of that intention?

Jev answers with a `noul` probability. GPT and Claude are asked the same question in ordinary sentences. The stored field is `plausibility_yes_probability`. Prompt version `v1` is the earlier JSON-only wording. Classification is a separate task: one distribution over `alarm_set`, `alarm_query`, and `alarm_remove`. One pairwise request carries one hypothesis and one utterance.

The default comparison is Jev, GPT, and Claude. A second hypothesis wording and a token log-probability baseline are in the tree and are not part of that comparison.

## Layout

- `data/massive/` — English MASSIVE, config `en-US`. Raw jsonl is gitignored. `checksum.json` is the dataset fingerprint.
- `data/pilot/` — 15 training originals, 60 draft inputs, meaning-change controls, the sampling manifest, and the review manifest.
- `prompts/` — versioned instructions, stored as sent. `prompts/v2/` asks GPT and Claude in sentences. `prompts/hypotheses/v2.json` is an alternate wording and is not the default.
- `configs/` — pinned model ids, endpoints, seed, and inference settings.
- `src/robust_likelihood/` — download, draft preparation, clients, runner, evaluation, and the log-probability module.
- `results/` — gitignored responses, metrics, and the disagreement list.
- `templates/supervisor_report.md` — write-up for run `20260927T151455Z`. A copy is stored with that run. The evaluator does not generate this file.

The 60 variants come from 15 underlying examples. Point metrics may use variant rows. They are not independent observations. A later confidence interval has to group variants by the original example.

## Keys

Prefer environment variables:

- `JEV_API_KEY` or `TYPESAFE_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

`API_key.txt` is an optional local fallback. It is gitignored, parsed by label, and never logged. Loading a key does not call the network. If a label is not recognized, the program stops and prints the label names only.

## Reproduce

```powershell
python -m pip install -e ".[dev]"
python -m robust_likelihood download
python -m robust_likelihood prepare
python -m pytest
python -m robust_likelihood run --dry-run
```

`download` fetches the official MASSIVE 1.1 archive, writes any missing split, and checks the published checksum. The official test split is not sampled.

`prepare` draws five train examples for each alarm intent with seed `20260927` and writes 60 draft inputs plus 15 meaning-change controls. `polite_restatement` is a transformation name in the schema only.

Review `data/pilot/review_manifest.json` before a real comparison. For each approved row, set `reviewer`, `reviewed_at`, `label_changed`, and `approved` to JSON `true`. Until then, a live run stops. `--allow-unreviewed` continues and marks the run `exploratory: true` in the manifest and in the results folder name.

## First comparison

Default models are pinned `jev-1.13.0`, `gpt-6-astra`, and `claude-fable-5-1`. `jev-latest` currently resolves to `jev-1.13.0`, and the comparison refuses an unpinned Jev id such as `jev-latest`. Connectivity may use an alias and does not start the comparison:

```powershell
python -m robust_likelihood connect --models jev --jev-model jev-latest
```

Jev is not sent a temperature. GPT-6 Astra accepts only its default temperature, and Claude Fable rejects `temperature`, so neither is sent one. Claude is allowed 16000 output tokens so adaptive thinking can finish before the JSON answer. The returned model id is stored beside the requested pin and does not replace the pin.

Baseline workload, one pair per request, before controls, repeats, and retries:

- Classification: 60 x 3 = 180
- Pairwise scoring: 60 x 3 x 3 = 540
- Total: 720

```powershell
python -m robust_likelihood run --dry-run
python -m robust_likelihood run --allow-unreviewed --max-requests 4
python -m robust_likelihood run --models jev,gpt,claude
python -m robust_likelihood run --hypothesis-version v2
python -m robust_likelihood evaluate results\<run-folder>
python -m robust_likelihood run --dataset synthetic --dry-run
```

`data/synthetic/alarm/` is a separate authored set. It is not sampled from MASSIVE, and its metrics are not pooled with the pilot. `--dataset synthetic` scores five intended-action hypotheses. State questions are stored beside those hypotheses and are not given truth labels. The default run remains the MASSIVE pilot.

`--dry-run` prints the planned workload and makes no API calls. `--max-requests` stops before exceeding that many new HTTP calls. Timeouts, a bounded retry count, and backoff are in `configs/models.json`. Successful responses are resumed from `results/cache.jsonl` and from `--resume`. Retry attempts share a cache key. `--repeats N` adds deliberate repeats with their own ids so the cache cannot collapse them.

The cache key is the pinned model id, task, input text, hypothesis id, hypothesis wording, hypothesis-description version, prompt version, and inference settings.

Every stored call includes the example ids, transformation, hypothesis, prompt version, requested and returned model ids, full response, parsed score or distribution, latency, usage, and error. Failures are stored.

## Evaluation

The runner evaluates as soon as the Jev–GPT calls finish. It does not renormalize pairwise scores into class probabilities.

- Classification: accuracy, negative log-likelihood, and multiclass Brier score.
- Pairwise: correct-intent rank, margin, and ranking flips.
- Meaning-preserving variants: paired score changes and yes/no prediction flips against the original wording, within model and hypothesis. Yes means probability above 0.5.
- Meaning-change controls: accuracy against the updated label.
- Coverage: attempted, succeeded, failed, and invalid counts for each model, beside the paired intersection.

A prediction is the unique highest score. A tie is the prediction `tie`, counts as incorrect, and is counted separately. Rank is 1 plus the number of alternatives with a strictly higher score. A tie for first place has margin 0. A ranking flip includes a unique winner becoming a tie, or the reverse.

Classification probabilities must lie in `[0, 1]` and sum to 1 within absolute tolerance `1e-3`. Otherwise the output is malformed and is not renormalized. Negative log-likelihood clips only the true-class probability to `[1e-6, 1]` before the log. A missing true-class probability is invalid. Rank, margin, and ranking flips require all three hypotheses. Paired model comparisons use items both models scored successfully.

`disagreements.jsonl` lists rows where Jev and GPT choose different unique predictions, or where one ties and the other does not. A fresh evaluation writes `inspection_notes` as an empty string. The notes for the four unchanged Jev–GPT rows in run `20260927T151455Z` were carried forward from the stored scores.

## Later steps

Token log-probability is `log P(e|h)` over observation tokens only. Calling it raises `NotImplementedError` with the message `not implemented`. It is not a completed baseline.

```powershell
python -c "from robust_likelihood.logprob import log_probability_evidence_given_hypothesis; log_probability_evidence_given_hypothesis(['set','an','alarm'], 'alarm_set')"
```
