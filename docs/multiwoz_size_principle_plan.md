# MultiWOZ size-principle test: are Jev's likelihoods correct?

Execution plan for Claude Code. Repository: `kaneis1/Robust-likelihood`, starting from `main` (commit `3119201`).
Save this file as `docs/multiwoz_size_principle_plan.md` and execute it phase by phase.

## 0. Purpose

Establish the paper's first claim: likelihood scores given by an LLM judge (Jev) are not likelihoods.

The criterion is the size principle (Bayesian Occam's razor). Suppose an utterance names a slot value. A hypothesis that constrains that slot must then beat a hypothesis with no preference about it, by a ratio we can measure from real users.

MultiWOZ supplies that reference:
- Every gold goal is a set of slots, so nested hypotheses can be generated automatically.
- The corpus gives two rates: P(first turn names slot X | user wants a value of X) and P(first turn names slot X | user has no preference about X).

**Pilot result being scaled up (Oct 25, one utterance).** Jev's breadth penalty depended on how breadth was worded, not on the breadth itself.
- Explicit "no preference about cuisine": logit gap 3.2 nats vs a corpus estimate of about 3.9.
- Implicit broad goal ("wants to eat somewhere"): 2.3 vs about 11.

**Deliverable.** One scatter plot of Jev log-ratio (y) against empirical log-ratio (x), with explicit and implicit wordings as different markers, plus `summary.md` with the metrics in Phase 6.

## 1. Rules for executing this plan

- Create branch `multiwoz-size-principle` from `origin/main`. Do not touch `gaussian-test`.
- Read these first and reuse their building blocks without changing their behavior: `README.md`, `src/robust_likelihood/{clients,planning,prompting,runner,storage,keys,massive}.py`, and `tests/`. If an existing module must change, keep the change minimal and its tests green.
- Preserve existing line endings (several files are CRLF). Do not reformat existing files.
- Python ≥ 3.11 and stdlib-first, like the rest of the repo. `matplotlib` (figures) and `torch`/`transformers` (Phase 8) are optional extras, imported lazily.
- API keys come only from the existing `keys.py` (`JEV_API_KEY` / `TYPESAFE_API_KEY`, or the gitignored `API_key.txt`). Never print, log, or commit a key.
- Items marked **[Frozen]** are pre-registered. To change one, write the reason in `docs/multiwoz_decisions.md` and ask the user first.
- **STOP and ask the user** at three points:
  1. Phase 2 counts differ from Appendix A.
  2. After the pilot, before the full Jev run.
  3. Before submitting the GPU job (Phase 8).
- Run `python -m pytest` and commit at the end of every phase.

## 2. Files to add

```
configs/multiwoz_size_principle.json
prompts/multiwoz_likely/{pairwise_question,classification_question,chat_pairwise,chat_classification}.txt
prompts/multiwoz_typical/{same four files}
src/robust_likelihood/multiwoz/__init__.py
src/robust_likelihood/multiwoz/__main__.py      # CLI: download, prepare, run, analyze, lm-score, report
src/robust_likelihood/multiwoz/download.py
src/robust_likelihood/multiwoz/corpus.py        # loading, selection, slot status
src/robust_likelihood/multiwoz/reference.py     # empirical ratios + bootstrap
src/robust_likelihood/multiwoz/hypotheses.py    # templates and controls
src/robust_likelihood/multiwoz/probe.py         # probe set assembly
src/robust_likelihood/multiwoz/jev_run.py       # Jev scoring via existing ModelClient
src/robust_likelihood/multiwoz/metrics.py
src/robust_likelihood/multiwoz/lm.py            # Phase 8 only (optional deps)
src/robust_likelihood/multiwoz/report.py        # summary.md + figures
scripts/multiwoz_lm_job.sh                      # GPU job template (Phase 8)
tests/fixtures/multiwoz_mini/                   # tiny hand-built 2.1 + 2.2 files
tests/test_multiwoz_*.py
data/multiwoz/checksum.json                     # committed
data/multiwoz/derived/{items.jsonl,probe.jsonl,reference_table.json,reference_table.md}   # committed
docs/multiwoz_decisions.md
```

Leave `src/robust_likelihood/logprob.py` untouched. Its test asserts `NotImplementedError`; Phase 8 code lives in `multiwoz/lm.py`.

## Phase 0: Scaffolding

1. In `pyproject.toml`, add these optional extras:
   - `figures = ["matplotlib>=3.8"]`
   - `lm = ["torch>=2.3", "transformers>=4.44", "accelerate>=0.33"]`
2. In `.gitignore`, add `data/multiwoz/raw/`.
3. Create `configs/multiwoz_size_principle.json`:

```json
{
  "multiwoz": {
    "repo": "budzianowski/multiwoz",
    "commit": "fe0c8e65cfcd8462bd33c86e35f21addc84ca82b",
    "zip_sha256": "8db3f6afe591383b8523b271f07727693e73012ecce47f3ac7987191e09b7523",
    "v22_dialogues_combined_sha256": "7eb537299a94838c245e7620ebd865d0237b86a69bc822685c210be1fb80bc87"
  },
  "slots": ["food", "area", "pricerange"],
  "other_domains": ["hotel", "train", "taxi", "attraction", "police", "hospital"],
  "stated_intention": "find_restaurant",
  "prompt_versions": ["multiwoz_likely", "multiwoz_typical"],
  "primary_set": "clean",
  "score_mapping": {"primary": "logit", "secondary": "log", "clip_epsilon": 0.001},
  "reference": {"a_population": "match_clean_turn0", "b_population": "pooled_first_restaurant_turn", "b_smoothing": 0.5},
  "bootstrap": {"replicates": 2000, "seed": 20261026},
  "pilot": {"items": 20, "set": "clean", "seed": 20261026},
  "repeat_check": {"items": 30, "repeats": 3, "prompt_version": "multiwoz_likely", "seed": 20261026},
  "lm": {"model": "Qwen/Qwen2.5-7B-Instruct", "samples_per_hypothesis": 64, "temperature": 1.0,
         "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0, "max_new_tokens": 96,
         "seed": 20261026, "bin_edges": [1.0, 2.0]}
}
```

Read the Jev model id from the existing `configs/models.json` (`jev-1.13.0`) rather than duplicating it.

## Phase 1: Download MultiWOZ

Command: `python -m robust_likelihood.multiwoz download`

1. Download over HTTPS with `urllib`, as `massive.py` does. Do **not** use `git clone`, because Windows `autocrlf` can rewrite text files and break the hashes. URL pattern:
   `https://raw.githubusercontent.com/budzianowski/multiwoz/<commit>/<path>`
2. Files to fetch, saved under `data/multiwoz/raw/`:
   - `data/MultiWOZ_2.1.zip`. Extract only `MultiWOZ_2.1/data.json`, `ontology.json`, `valListFile.txt`, and `testListFile.txt`.
   - `data/MultiWOZ_2.2/train/dialogues_001.json` … `dialogues_017.json`
   - `data/MultiWOZ_2.2/dev/dialogues_001.json`, `dialogues_002.json`
   - `data/MultiWOZ_2.2/test/dialogues_001.json`, `dialogues_002.json`
   - `data/MultiWOZ_2.2/schema.json`
   - Skip `dialog_acts.json` (60 MB, not needed).
3. Write each file to a temporary name, then rename it into place.
4. Verify hashes:
   - The zip's sha256 must equal `zip_sha256`.
   - Combined 2.2 digest: take the 21 `dialogues_*.json` files, sort their paths relative to `MultiWOZ_2.2/` (`dev/…`, `test/…`, `train/…`), concatenate their lowercase hex sha256 strings with no separator, and take the sha256 of that ASCII string. It must equal `v22_dialogues_combined_sha256`.
5. Write `data/multiwoz/checksum.json` with the commit, per-file sha256, the combined digest, and a timestamp. Re-running must not re-download files whose hashes already match.
6. Acceptance:
   - `data.json` has 10,438 dialogues.
   - The 2.2 files contain 10,437 dialogues.
   - Unit test: hash verification against fixture bytes with a fake downloader (no network).

## Phase 2: Parse and select (`prepare`, part 1)

Definitions **[Frozen]**:

- **Restaurant-only:** `goal["restaurant"]` is non-empty and every domain in `other_domains` is empty.
- **Analysis population:** dialogue id starts with `SNG` and the dialogue is restaurant-only. This gives 442 dialogues.
- **First-turn goal:** `G_first = goal.restaurant.info` updated by `goal.restaurant.fail_info` (fail_info wins). Rationale: fail_info holds the constraints the user starts with before falling back. The 2.2 turn-0 annotation agrees with fail_info in most cases.
- **Name-based goals:** `"name"` in `G_first`. Exclude these (72), because they have no slot hypotheses.
- **Constrained slots:** `C = {s in slots : G_first.get(s) not in (None, "dontcare")}`.
- **Set:**
  - `clean` if `fail_info` is empty, otherwise `fail_info`.
  - **Primary analysis = clean.** A wrong goal value would make the narrow hypothesis look contradicted and inflate the effect being tested, so excluding fail_info items is conservative.
- **Turn 0:** `turns[0]` of the 2.2 dialogue (assert the speaker is `USER`). Take the utterance from 2.2. Compare it with 2.1 `log[0].text` after normalizing whitespace and case, and report mismatches without failing.
- **Slot status in a turn.** Read the restaurant frame's `state.slot_values["restaurant-<slot>"]`, which is a list of surface variants:
  - `named`: non-empty and not containing `"dontcare"`
  - `dontcare`: contains `"dontcare"`
  - `omitted`: otherwise

  Turn 0 uses the state directly. Later turns use the delta against the previous USER turn's restaurant state.
- **Match:** the lowercased goal value appears in the lowercased value list. Record mismatches.
- **First restaurant turn** (pooled population only): the first USER turn whose restaurant frame has `active_intent != "NONE"` or non-empty `slot_values`.

Output `data/multiwoz/derived/items.jsonl` (370 rows), one row per slot-based analysis dialogue, with these fields:
`item_id, set, utterance, goal (constrained slots only), n_constrained, slot_status{food,area,pricerange}, slot_match, has_fail_info, utterance_matches_v21`

**Acceptance:** must equal Appendix A, table A1, exactly. Otherwise STOP.

## Phase 3: Reference ratios (`prepare`, part 2)

Populations:

| Name | Turns | Size |
|---|---|---|
| `match_clean_turn0` | turn 0 of the 235 clean items | 235 |
| `match_all_turn0` | turn 0 of all 370 slot-based items | 370 |
| `pooled_first_restaurant_turn` | first restaurant turn of every dialogue (any prefix) with a slot-based restaurant goal | 3,927 |
| `pooled_no_woz` | as above, excluding `WOZ*` dialogues | (record) |

Estimators, per slot X:

- `a_X = (#turns naming X) / (#dialogues with X constrained)`, unsmoothed, with a Wilson 95% CI.
- `b_X = (#turns naming X + 0.5) / (n + 1)`, where n counts dialogues with X absent **or** `dontcare` in the goal. These are both "no preference".
- Per-pair reference terms:
  - X named in the turn: `ln(a_X / b_X)`
  - X omitted: `ln((1 − a_X) / (1 − b_X))`
- Bootstrap, B = 2000, resampling dialogues within each population. Store all replicates so Phase 6 can combine them.

**Primary [Frozen]:** take `a_X` from `match_clean_turn0` and `b_X` from `pooled_first_restaurant_turn`.

Why mix two populations:
- `a` must match the probe population. It is about 0.73 at turn 0 of single-domain dialogues, but about 0.56 at the first restaurant turn of the pooled corpus.
- `b` is a rare-event rate. The matched population has only 0 or 1 events per slot, so it needs the pooled corpus.

Sensitivity variants, reported in the same table:

| Variant | a from | b from |
|---|---|---|
| V2 | match_all_turn0 | match_all_turn0 |
| V3 | pooled | pooled |
| V4 | pooled_no_woz | pooled_no_woz |

Also report per-slot value-mismatch rates. These give context for the contradiction control; no formal ratio is needed.

The slot-level ratio is **conservative**. It ignores which value was named, and under a no-preference goal a specific value is even less likely. So the true likelihood ratio is at least `a/b`. Say this in the report.

Outputs: `data/multiwoz/derived/reference_table.json` and `.md`.

**Acceptance:** must match Appendix A, tables A2 and A3 (±0.01 on rates, exact on counts).

## Phase 4: Hypotheses and probe set (`prepare`, part 3)

### Templates [Frozen]

```
PRICE = {"cheap": "cheap", "moderate": "moderately priced", "expensive": "expensive"}
NAMES = {"food": "the type of food", "area": "the area", "pricerange": "the price range"}

base(g):  "The user wants to find {a|an} [PRICE ]restaurant[ serving {food} food][ in the {area} of town]."
          (use "an" when the next word starts with a vowel; food values verbatim from the goal, lowercase)

explicit(g): base(g) + " The user has no preference about {list}."
             list = NAMES of slots NOT in g, in the order food, area, pricerange,
             joined as "A", "A or B", "A, B, or C". If every slot is in g, explicit(g) == base(g).
```

Both regimes share the first sentence word for word. The **only** difference is the explicit "no preference" sentence. This isolates the wording effect.

### Hypotheses per item (keys are internal and never sent to Jev)

| Key | Text | When |
|---|---|---|
| `G|impl` | base(G) | always |
| `G|expl` | explicit(G) | only if \|C\| = 2; for \|C\| = 3 reuse the `G|impl` score for both regimes |
| `minus:X|impl` / `minus:X|expl` | base / explicit of G without X | each X in C |
| `broad|impl` / `broad|expl` | base({}) / explicit({}) | always |
| `ctrl:contra` | base(G with one **named** slot changed) | always (see below) |
| `ctrl:tie` | base(G with one **omitted** constrained slot changed) | if some slot in C is omitted |

### Controls

- **Slot choice and new value:** seeded with `int(sha256(f"{item_id}:{kind}").hexdigest()[:8], 16)`.
- **Value pools:**
  - food: every food value seen in any restaurant goal (`info` ∪ `fail_info`), minus `dontcare`
  - area: centre, north, south, east, west
  - price: cheap, moderate, expensive

  Never reuse the goal value.
- **No named constrained slot:** if no slot in C is named (11 items), make `ctrl:contra` a second tie control, keyed `ctrl:tie2`.

### Output

`data/multiwoz/derived/probe.jsonl`, one row per (item, hypothesis), with fields:
`item_id, set, utterance, hyp_key, hyp_type, regime, constrained, removed, text`

Print the number of distinct texts. Expect about 700 non-control texts.

### Acceptance

- 3,727 rows in total (clean 2,355; fail_info 1,372).
- Unit tests:
  - Regime texts differ only by the explicit sentence.
  - No hypothesis key or type ever appears in a text.
  - Articles are correct.
  - The \|C\| = 3 `G|expl` row is deduplicated.
  - Each control differs from G in exactly one slot.
  - Seeding is deterministic.

### Example (2-slot goal; utterance "I am looking for an expensive Italian restaurant.")

```
G|impl        The user wants to find an expensive restaurant serving italian food.
G|expl        The user wants to find an expensive restaurant serving italian food. The user has no preference about the area.
minus:food    The user wants to find an expensive restaurant.  [+ " The user has no preference about the type of food or the area."]
minus:price   The user wants to find a restaurant serving italian food.  [+ " The user has no preference about the area or the price range."]
broad         The user wants to find a restaurant.  [+ " The user has no preference about the type of food, the area, or the price range."]
```

## Phase 5: Jev scoring (`run`)

### Prompt packs [Frozen]

Copy `prompts/likely/*` into `prompts/multiwoz_likely/` and `prompts/typical/*` into `prompts/multiwoz_typical/`. Then replace only `pairwise_question.txt` with these texts:

- `multiwoz_likely`:
  `Assume the user has the stated intention described in the state. The utterance is the user's first message to a restaurant booking assistant. Is the user likely to produce the following utterance when expressing that intention? Utterance: "<<UTTERANCE>>"`
- `multiwoz_typical`:
  `Assume the user has the stated intention described in the state. The utterance is the user's first message to a restaurant booking assistant. Is the following utterance typical of what a user would say when expressing that intention? Utterance: "<<UTTERANCE>>"`

### Request

Use one hypothesis per call so scores are per-goal likelihoods. Reuse `clients.build_jev_request` through `ModelClient.perform` with a `planning.CallSpec`:

- `task="pairwise"`, `provider="jev"`
- `hypothesis="find_restaurant"`: a **constant [Frozen]**, never the hypothesis key, so the label cannot leak breadth
- `hypothesis_wording=<text>`, `input_text=<utterance>`, `prompt_version=<pack>`
- `dataset="multiwoz_size"`, `example_id=item_id`, `pair_id=item_id`, `role=hyp_key`
- `transformation="none"`, `intent=original_intent="find_restaurant"`, `label_changed=False`
- `hypothesis_description_version="multiwoz_v1"`, `inference_settings={}`
- The `HypothesisSet` argument is unused on this path; pass `HypothesisSet("multiwoz_v1", {})`.

The resulting body is:
`{"model", "state": {"utterance", "stated_intention": "find_restaurant", "intention_description": text}, "questions": {"plausibility": {"type": "noul", "instructions": ...}}}`

The score is `answers.plausibility.noul`.

### Caching and outputs

- Cache: `results/multiwoz/cache.jsonl`, keyed by `planning.cache_key(spec)`. Successful records are reused across runs.
- Run directory: `results/multiwoz/<UTC>-<label>/`, containing:
  - `manifest.json`: git commit, config, prompt texts, requested and returned model ids, counts, failures, timing
  - `calls.jsonl`: raw records, same schema as `runner._record`
  - `scores.jsonl`: `item_id, hyp_key, prompt_version, repeat_id, noul`
- Abort if the returned model id changes mid-run.

### CLI

```
python -m robust_likelihood.multiwoz run --wording likely|typical|both --set clean|fail_info|all
    [--items N --seed S] [--repeats R] [--max-requests N] [--dry-run] [--resume DIR] [--label NAME]
```

`--dry-run` prints the call count and the first three request bodies. It makes no HTTP calls; test this with a fake transport.

### Order

1. `python -m robust_likelihood connect --models jev` to check credentials (existing command).
2. **Pilot:** 20 seeded clean items, both wordings (about 400 calls). Write `pilot_report.md` with:
   - success rate (must be ≥ 99%)
   - returned model id (must be constant)
   - median ℓ(`G|impl`) − ℓ(`ctrl:contra`) (expect > 2)
   - a table for 5 items in the layout of the Oct 25 run

   Then **STOP and ask the user.**
3. **Full run:** all 370 items × 2 wordings = **7,454 calls**. At about 0.5 s per call, run sequentially, this takes about 1 to 1.5 hours.
4. **Repeat check:** 30 clean items × 3 repeats on `multiwoz_likely` (about 900 calls) to measure Jev's within-item noise. If the repeats are identical, report that Jev is deterministic.

## Phase 6: Metrics (`analyze`)

All definitions here are **[Frozen]**.

**Score mapping.**
- Primary: ℓ = logit(clip(s, ε, 1 − ε)) with ε = 1e-3. This was used in the pilot, and it is the generous reading for Jev: logit gaps ≥ log gaps when the narrow hypothesis scores higher, so any shortfall is conservative.
- Secondary: λ = ln(clip(s)).
- Report the number of clipped scores.

**Pairs, within regime r ∈ {expl, impl}:**
- Single-slot removal: (`G|r`, `minus:X|r`) for each X ∈ C. Its direction is X's turn-0 status: named or omitted.
- Full removal: (`G|r`, `broad|r`), grouped by (|C|, k_named).

**Jev log-ratio:** L_J = ℓ(narrow) − ℓ(broad).

**Empirical log-ratio:** L_E = Σ over removed X of [named ? ln(a_X/b_X) : ln((1−a_X)/(1−b_X))].

**Occam gap (headline).** OG = mean(L_J) − L_E, per group of:
prompt version × regime × removal type (food, area, price, or all) × direction (named, omitted, or mixed).

CIs come from paired bootstrap replicates: replicate b resamples items for L_J and uses reference replicate b for L_E.

**Wording gap.** Δ = ℓ(expl) − ℓ(impl) for the same breadth (`minus:X`, `broad`, and `G` for |C| = 2). Report the mean, CI, and share with |Δ| > 1. The target is 0.

**Monotonicity violations.** Rate of ℓ(broad) > ℓ(narrow), strict, among single-slot pairs whose removed slot is named, and among full-removal pairs whose constrained slots are all named. Report ties separately.

**Controls.**
- Contradiction: AUROC of ℓ(`G|impl`) vs ℓ(`ctrl:contra`), plus the mean gap by changed slot.
- Tie offset (the H5-type bias): mean ℓ(`G|impl`) − ℓ(`ctrl:tie`). The target is 0.

**Per-hypothesis bias.** Within-item centred ℓ, averaged by `hyp_type`.

**Reverse direction (Phase 7).** OG restricted to omitted-slot pairs. Expect L_E of about −1.3 nats per slot.

**Secondary analyses.**
- the fail_info set
- the λ mapping
- reference variants V2 to V4
- repeat noise

**Outputs:** `metrics.json` and markdown tables.

### Pre-registered predictions (from the pilot)

Failing a prediction is a result, not a bug. Report a pass/fail scorecard.

| ID | Prediction |
|---|---|
| P1 | Explicit, named, single-slot OG within ±1.5 nats (mean over slots, `likely`) |
| P2 | Implicit, named, single-slot OG ≤ −2 nats |
| P3 | Implicit full removal with all slots named: OG ≤ −6 nats |
| P4 | Wording gap mean ≤ −1 logit (explicit penalized more than implicit) |
| P5 | Tie offset \|mean\| > 0.5 logit (per-hypothesis bias present) |
| P6 | Phase 8: LM log-likelihood named single-slot L ≥ +2 nats; both typicality scores have \|L\| ≤ 0.5 nats or negative |

## Phase 7: Reverse direction

This needs no extra calls. The omitted-slot pairs come from the same probe set: 82 to 87 per slot across all 370 items, and about 50 per slot in the clean set. Here the broader hypothesis should win by `ln((1−b)/(1−a))`.

An optional extension (do not run unless asked) would add the first restaurant turns of multi-domain dialogues. Those turns are often not self-contained ("same area as the hotel"), so one-off scoring would be invalid for them.

## Phase 8: Normalized-typicality check (GPU, `lm-score`)

Run this on a GPU node with ≥ 24 GB, bf16:
`python -m robust_likelihood.multiwoz lm-score --probe data/multiwoz/derived/probe.jsonl`
It writes `lm_scores.jsonl` into the run directory.

### Model

`Qwen/Qwen2.5-7B-Instruct`. Record the exact Hugging Face revision hash in the manifest.

### Prompt [Frozen]

- system: `You simulate users of a restaurant booking assistant.`
- user: `Description of the user: {hypothesis text}\nWrite this user's first message to the assistant. Reply with the message only.`
- assistant: `{utterance}`

### Steps

1. **Surprisal.** S(u|h) = −Σ log p over the assistant tokens **plus** the end-of-turn token. Build the prefix with the chat template (`add_generation_prompt=True`), and score only the tokens after the prefix.
2. **Reference generator.** For each distinct hypothesis text, draw K = 64 samples from the same prefix with `do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=96`.
   - Override the model's generation_config **explicitly**. Qwen's defaults (top_p 0.8, top_k 20, repetition penalty 1.05) would sample from a different distribution than the one being scored.
   - Drop samples that hit `max_new_tokens` without end-of-turn, and report that rate.
   - Compute each hypothesis's mean μ_h and standard deviation σ_h of S over its samples.
3. **Scores.**
   - `lm_loglik = −S`: the "exact" analogue
   - T = (S − μ_h)/σ_h, and `lm_typ_normal = −T²/2`
   - `lm_typ_bins`: ln of the empirical mass of |T|'s bin (edges 1 and 2) among h's own samples, with add-one smoothing: (count + 1)/(K + 3)
4. **Analysis.** Rerun Phase 6 with L = score(narrow) − score(broad). These scores are already in nats, so no logit.
5. **Diagnostics.** Plot μ_h and σ_h against |C|. The mechanism to show is that μ_h rises as hypotheses broaden. Also report the explicit-vs-implicit wording gap for the LM.

### Cost and the stop point

There are about 1,000 to 1,300 distinct texts. At 64 samples each, about 30 tokens per sample, that is roughly 2 to 3 M generated tokens, under an hour on one A100.

Write `scripts/multiwoz_lm_job.sh` with placeholders for LSF (`bsub`) and PBS (`qsub`). **STOP and ask** before submitting it.

## Phase 9: Figures and report (`report`)

### Headline figure

Save as `fig_occam_<wording>.png` (dpi 200), plus `.pdf` and the plotted data as `.csv`.

- One point per group:
  - single-slot named (food, area, price)
  - single-slot omitted (same three)
  - full removal by (|C|, k_named), merging groups with fewer than 15 items
- x = L_E (primary reference) with its 95% CI; y = mean L_J with its 95% CI.
- Markers: explicit as filled circles, implicit as open squares. Colour by group type.
- Draw a dashed y = x diagonal and zero lines.
- Also produce a two-panel version, `likely` and `typical`.

### Supplementary figures

- per-item hexbin of L_J vs L_E
- score histograms by `hyp_type`
- the same headline layout for the LM scorers
- μ_h vs |C|

### `summary.md` sections

1. Setup and counts
2. Reference table
3. Headline OG (explicit vs implicit, likely vs typical)
4. Wording gap
5. Monotonicity
6. Controls and tie offset
7. Reverse direction
8. Typicality and LM
9. Prediction scorecard
10. Notes for Xuan

In "Notes for Xuan", state the result on whether normalization cancels Occam's razor, using the μ_h-vs-breadth diagnostic.

## Phase 10: Tests and pull request

Tests run on `tests/fixtures/multiwoz_mini/`, with no network and no GPU:

- **download:** digest functions, idempotency
- **corpus:** selection, `G_first`, slot status, first restaurant turn, delta
- **reference:** estimators, smoothing, bootstrap determinism
- **hypotheses/probe:** the Phase 4 tests, plus row counts on the fixture
- **jev_run:** body shape, constant `stated_intention`, prompt text, dry-run makes no calls, cache hit on rerun
- **metrics:**
  - an oracle scorer whose scores follow the reference likelihoods gives OG ≈ 0
  - a constant scorer gives OG = −L_E
  - identical texts give a wording gap of 0
  - AUROC on known inputs
- **lm:** skipped unless `torch` is installed. Assert that only assistant tokens are scored.

The full existing suite must still pass. Open a PR to `main` with `summary.md` and the headline figure attached.

## Appendix A: Expected numbers (checked on the pinned data)

**A1. Selection** (MultiWOZ 2.1 goals + 2.2 turn-0 annotations)

| Quantity | Count |
|---|---|
| Dialogues in 2.1 `data.json` / in 2.2 | 10,438 / 10,437 |
| With a restaurant goal / restaurant-only / SNG restaurant-only | 4,692 / 1,311 / **442** |
| Name-based (excluded): clean / fail_info | 44 / 28 (total 72) |
| Slot-based: clean / fail_info | **235** / 135 (total 370) |
| Clean items by \|C\|: 2 slots / 3 slots | 124 / 111 |
| Items with no named constrained slot | 11 |

Clean set, turn-0 slot status:

| Slot | Constrained | Named (of which mismatch) | Omitted | No-pref | No-pref named |
|---|---|---|---|---|---|
| food | 195 | 142 (0) | 53 | 40 | 2 |
| area | 186 | 138 (3) | 48 | 49 | 0 |
| pricerange | 200 | 150 (1) | 50 | 35 | 1 |

**A2. Pooled first restaurant turn** (3,927 dialogues with a slot-based restaurant goal)

| Slot | Named / constrained | No-pref named / n (absent + dontcare goals) |
|---|---|---|
| food | 1,809 / 3,108 | 5 / 819 |
| area | 1,730 / 3,104 | 12 / 823 |
| pricerange | 1,758 / 3,167 | 10 / 760 |

Excluding WOZ, a is about 0.50 to 0.53 and the no-pref named counts are 3/567, 8/559, and 8/498.

**A3. Primary reference** (a from clean turn 0, b pooled with +0.5 smoothing)

| Slot | a | b | Named: ln(a/b) | Omitted: ln((1−a)/(1−b)) |
|---|---|---|---|---|
| food | 0.728 | 0.00671 | 4.69 | −1.30 |
| area | 0.742 | 0.01517 | 3.89 | −1.34 |
| pricerange | 0.750 | 0.01380 | 4.00 | −1.37 |

All three slots named and fully removed: L_E = 12.57 nats.

**A4. Calls**

| | Count |
|---|---|
| Probe rows per prompt version (clean / fail_info) | 3,727 (2,355 / 1,372) |
| Jev calls for both versions | 7,454 |
| Pilot | about 400 |
| Repeat check | about 900 |

## Appendix B: Command sequence

```
git checkout -b multiwoz-size-principle origin/main
python -m pip install -e ".[dev,figures]"
python -m robust_likelihood.multiwoz download
python -m robust_likelihood.multiwoz prepare            # Phases 2–4; prints Appendix A checks
python -m pytest
python -m robust_likelihood connect --models jev
python -m robust_likelihood.multiwoz run --wording both --set clean --items 20 --label pilot   # then STOP
python -m robust_likelihood.multiwoz run --wording both --set all --label full
python -m robust_likelihood.multiwoz run --wording likely --set clean --items 30 --repeats 3 --label repeats
python -m robust_likelihood.multiwoz analyze results/multiwoz/<full-run>
python -m robust_likelihood.multiwoz lm-score ...       # GPU node, after approval
python -m robust_likelihood.multiwoz report results/multiwoz/<full-run>
```
