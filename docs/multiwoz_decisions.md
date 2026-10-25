# MultiWOZ size-principle decisions

The plan is `docs/multiwoz_size_principle_plan.md`. Changes to **[Frozen]** items are
recorded here with the reason, and only after the user agrees. Interpretations of
points the plan leaves open are recorded too, so the analysis can be audited.

## Environment

- The macOS machine had only the system Python 3.9. The repo needs 3.11 or later, so a
  Python 3.12 virtual environment was made with `uv` at `~/.venvs/robust-likelihood`,
  outside OneDrive, and the package was installed editable into it.
- Existing CRLF files keep CRLF when edited (`pyproject.toml`, `.gitignore`). New files
  are LF, as in the formal typicality commit. The copied prompt packs keep the CRLF
  bytes of their sources.

## Phase 2: one Appendix A1 cell differs

`prepare` reproduces every A1 count except one. Clean set, food, "No-pref named":
Appendix A1 says 2; the frozen status rule gives 1.

- The second clean no-preference item is SNG0512, "Hi! I'm looking for a cheap restaurant
  in the north, any type!" Its turn-0 food value is `["dontcare"]`. The frozen rule makes
  that `dontcare`, not `named`, so it is reported as `no_pref_dontcare: 1`.
- The same rule reproduces every pooled A2 count exactly: 3,927 dialogues; food 1,809/3,108
  and 5/819; area 1,730/3,104 and 12/823; price 1,758/3,167 and 10/760; excluding WOZ,
  3/567, 8/559, and 8/498. Counting `dontcare` as named would instead give no-preference
  counts of 11/819, 36/823, and 26/760, which contradict A2 and A3.
- The primary reference does not use this cell: `a` comes from constrained clean items and
  `b` from the pooled population. Only the V2 sensitivity variant (both rates from
  `match_all_turn0`) uses it: food `b` is 1.5/63 under the frozen rule, 2.5/63 otherwise.

Decision, 26 October 2026: the user chose to keep the frozen rule. The A1 check in
`corpus.APPENDIX_A1` now expects food `no_pref_named: 1` and `no_pref_dontcare: 1`. The
plan document keeps its original table.

## Phase 3: where the bootstrap replicates live

The 2,000 replicates of every population are stored as integer counts
(`named, constrained, no_pref_named, no_pref` per slot) in
`data/multiwoz/derived/reference_replicates.json`, written compactly (about 400 KB) so
`reference_table.json` stays readable. `reference_table.json` records that file's sha256.
Each population's resample is drawn from
`random.Random(int(sha256(f"{seed}:{population}").hexdigest()[:16], 16))` over rows in
dialogue-id order, so Phase 6 can redraw the clean-item resample of replicate b and pair
Jev's scores with the a-rates of the same resampled items.

## Phase 4: points the plan leaves open

- Keys use the slot names: `minus:pricerange|impl`, not the plan example's `minus:price`.
  Figures and tables label that slot "price".
- `hyp_type` is the key without its regime (`G`, `minus:food`, `broad`, `ctrl:contra`, ...).
  Controls are rendered with `base()`, so their regime is `impl`.
- Probe rows carry two fields beyond the plan's list: `values`, the slot values the text
  states, and, on controls, `changed` (slot, old value, new value). Both make the
  one-slot-difference check auditable without the raw data.
- For the 11 items with no named constrained slot, `ctrl:tie2` uses its own seed
  (`{item_id}:tie2`) but skips the slot and value that `ctrl:tie` already chose, so the
  second tie control is never a copy of the first. All 11 `tie2` texts differ from `tie`.
- No goal in the corpus has food `dontcare`, so the food pool is 91 values.
- The probe has 3,727 rows (clean 2,355; fail_info 1,372) and 707 distinct non-control texts.
