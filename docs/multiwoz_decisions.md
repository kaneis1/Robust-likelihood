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
