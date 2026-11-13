# Simulator study decisions

The plan is `docs/simulator_study_plan.md`. Changes to **[Frozen]** items are recorded here with
the reason, and only after the user agrees. Interpretations of points the plan leaves open are
recorded too, so the analysis can be audited.

## Environment

- Branch `simulator-study` from `origin/claims-123` at `d8d3e0e`.
- Models are downloaded on the SoC login node with `huggingface_hub` 0.36.2 in a throwaway
  environment, pinned to the revisions below, skipping `.bin`, `.pth`, `.pt`, `.gguf`,
  `original/`, and `onnx/` copies of the safetensors weights. Jobs run with `HF_HUB_OFFLINE=1`.

## Models (checked on the Hugging Face Hub, 2026-11-13)

| Key | Id | Revision | License | Architecture |
|---|---|---|---|---|
| `qwen25_inst`, `qwen25_broad` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c70…` | apache-2.0 | Qwen2ForCausalLM |
| `qwen3_base` | `Qwen/Qwen3-8B-Base` | `49e3418fbbbc…` | apache-2.0 | Qwen3ForCausalLM |
| `qwen3_inst` | `Qwen/Qwen3-8B` | `b968826d9c46…` | apache-2.0 | Qwen3ForCausalLM |
| `osim_mid` | `cmu-lti/osim-8b-mid` | `af9ef0347ede…` | mit | Qwen3ForCausalLM |
| `osim_final` | `cmu-lti/osim-8b` | `a0002c6686e7…` | mit | Qwen3ForCausalLM |
| `userlm` | `microsoft/UserLM-8b` | `ce576a239996…` | mit | LlamaForCausalLM |

Full hashes are in `configs/simulator.json`. `cmu-lti/osim-8b` resolves, so the `-post`
fallback is not needed.

## Open for the user (Phase 1 stop)

1. **OSim lineage.** The `cmu-lti/osim-8b` card lists `Qwen/Qwen3-8B` (the instruct model) as its
   base: Qwen3-8B midtrained on OdysSim, then task RL and expert consolidation. `osim-8b-mid` is
   `Qwen3-8B-Base` midtrained. The final model is therefore not the mid checkpoint plus RL, and
   mid vs final (Q5, S2) mixes the starting point (base vs instruct) with RL. Cleaner contrasts
   the four models allow: `qwen3_base` → `osim_mid` (midtraining from base) and `qwen3_inst` →
   `osim_final` (midtraining plus RL from instruct).
2. **`osim_inst` (optional).** `sunweiwei/OSim-Inst-8B` is a vision-language model
   (`Qwen3VLForConditionalGeneration`), lists no license, and has an empty card. Proposed: leave it
   out.
3. **Format S for UserLM.** UserLM's chat template only ever opens a *user* turn (it appends the
   user header after the messages, whatever `add_generation_prompt` says), because the model is
   trained to speak as the user. S as written puts U in the assistant role, which UserLM's template
   cannot even produce. Implemented: the simulated human speaks in the role the model generates,
   and the opener takes the other role, so for UserLM the opener is an assistant message and U a
   user turn. For every other model this is S exactly as written.
4. **Few-shot examples (format F).** The plan asks for 2- and 3-slot goals and one turn that omits
   a constrained slot. A 3-slot first turn that names all three slots is almost absent from MultiWOZ
   (1 of 228 train candidates; none of the 111 clean 3-slot probe items). Implemented: 4 two-slot
   and 4 three-slot turns drawn at random within each size (seed 20261103), so naming and omission
   follow the corpus; 5 of the 8 omit a constrained slot. `data/multiwoz/derived/fewshot_examples.json`.

## Interpretations (no change to a frozen item)

- **Prefix and boundary per model.** Prefix = the chat template's generation prompt, with the
  model's template options, plus the separator the template puts before a turn's content (none
  for Qwen and OSim; a newline for UserLM, whose header has none). Reply = U, then the end-of-turn
  token (`<|im_end|>` for Qwen and OSim, `<|eot_id|>` for UserLM). A template check confirms the
  template closes the simulated human's turn with that token.
- **Thinking.** `Qwen3-8B` takes `enable_thinking=False`, and its prefix includes the empty think
  block the template inserts, the same for every hypothesis. The OSim templates have no thinking
  handling, so no option is passed. Every sample is checked for a non-empty `<think>` block.
- **UserLM generation.** The card samples with `top_p=0.8` and bans `<|endconversation|>` through
  `bad_words_ids`, a logits processor. The plan's pure sampling and the exact-sampling check rule
  both out, so neither is used; samples containing `<|endconversation|>` are flagged. No BOS token
  is added, as in the card's own `apply_chat_template` usage.
- **N for OSim.** The cards describe generating the next human turn given the other party's turns;
  whether OSim writes a sensible first turn from the system prompt alone is what Phase 1 shows.
- **Format F.** Sampling stops at the first newline (a stopping rule, which leaves the
  distribution exact; the exactness check runs without it), or at `<|endoftext|>`. A sample's own
  surprisal is its text rescored as " {text}\n", since the newline can fall inside a token; chat
  samples are scored on their own token ids, as Phase 8 scored its samples.
- **Format check.** 3 two-slot and 3 three-slot clean items (seed 20261103), their `G|impl`
  texts; 8 samples each; the boundary check on 200 seeded clean probe pairs per (model, format);
  P0 against Phase 8 on 20 seeded pairs.
