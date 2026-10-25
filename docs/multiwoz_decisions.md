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

## Phase 2: one Appendix A1 cell differs (awaiting the user)

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

Recommendation: keep the frozen rule and read the A1 cell as 1 named plus 1 `dontcare`.
Not acted on until the user agrees.
