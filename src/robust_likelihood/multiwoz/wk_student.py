"""Week of 24 Nov, Part D (docs/week_nov24_plan.md 5.4): train a small typicality student, then predict.

Backbone Qwen/Qwen3-0.6B-Base at its pinned revision (configs/simulator.json week_nov24.student), with
LoRA r = 16 on every linear layer of the transformer (written here: the cluster environment has no
peft, and jobs never install). Input: "Customer description: {h}\nCustomer message: {x}" and the end
token; the head reads the last token's final hidden state.

  R-lin  linear head, MSE on the clipped T
  R-mlp  MLP head (one hidden layer of 512, GELU), MSE on the clipped T
  C-mlp  MLP head, cross-entropy on the bins

LoRA learning rate 2e-4, head 1e-3, AdamW, cosine decay to 0 with no warmup, batch 64, 2 epochs, bf16
autocast on the GPU with fp32 LoRA and head weights. The validation loss is read every eval_every
steps and at the end; the best state is kept and used for prediction. After training the run
predicts every evaluation pair: the human pairs, the claims utterance and hypothesis rewrites, Part
A's units, and the synthetic validation and held-out-food pairs, and times pairs per second.
torch and transformers are imported inside the functions only.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from robust_likelihood.multiwoz import DERIVED_DIR, RESULTS_DIR
from robust_likelihood.multiwoz.jev_run import git_state
from robust_likelihood.storage import read_json, read_jsonl, write_json, write_jsonl

RUNS = {"R-lin": {"head": "linear", "loss": "mse"}, "R-mlp": {"head": "mlp", "loss": "mse"},
        "C-mlp": {"head": "mlp", "loss": "ce"}}
PROMPT = "Customer description: {h}\nCustomer message: {x}"
SETTINGS = {"lora_r": 16, "lora_alpha": 32, "lr_lora": 2e-4, "lr_head": 1e-3, "batch_size": 64, "epochs": 2,
            "hidden": 512, "max_tokens": 256, "eval_every": 1000, "seed": 20261124}


def encode(tokenizer, h: str, x: str, max_tokens: int) -> list[int]:
    ids = tokenizer(PROMPT.format(h=h, x=x), add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return list(ids)[: max_tokens - 1] + [tokenizer.eos_token_id]


def build_student(backbone, hidden_size: int, head: str, outputs: int, settings: dict):
    """Freeze the backbone, wrap its linear layers in LoRA, and add the head."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = alpha / rank

        def forward(self, x):
            return self.base(x) + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    targets = [(parent, name, child) for parent in backbone.modules() for name, child in parent.named_children()
               if isinstance(child, nn.Linear)]
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, settings["lora_r"], settings["lora_alpha"]))

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            if head == "linear":
                self.head = nn.Linear(hidden_size, outputs)
            else:
                self.head = nn.Sequential(nn.Linear(hidden_size, settings["hidden"]), nn.GELU(),
                                          nn.Linear(settings["hidden"], outputs))

        def forward(self, ids, mask):
            states = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            last = states[torch.arange(ids.shape[0], device=ids.device), mask.sum(dim=1) - 1]
            return self.head(last.float())

    student = Student().to(next(backbone.parameters()).device)
    return student, len(targets)


def batches(sequences: list[list[int]], order: list[int], size: int, pad: int, device):
    import torch

    for start in range(0, len(order), size):
        chunk = order[start:start + size]
        width = max(len(sequences[index]) for index in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, index in enumerate(chunk):
            ids[row, :len(sequences[index])] = torch.tensor(sequences[index], dtype=torch.long)
            mask[row, :len(sequences[index])] = 1
        yield chunk, ids.to(device), mask.to(device)


def train_student(student, train: list[dict], val: list[dict], spec: dict, settings: dict, pad: int, device,
                  log=print) -> dict:
    """Train in place; returns the curve and keeps the best validation state in the student."""
    import torch

    loss_kind = spec["loss"]
    autocast = device.type == "cuda"
    lora = [parameter for name, parameter in student.named_parameters() if parameter.requires_grad
            and not name.startswith("head.")]
    head = list(student.head.parameters())
    optimizer = torch.optim.AdamW([{"params": lora, "lr": settings["lr_lora"]},
                                   {"params": head, "lr": settings["lr_head"]}])
    steps_per_epoch = math.ceil(len(train) / settings["batch_size"])
    total = steps_per_epoch * settings["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total)
                                                                                                / total)))

    def loss_of(output, rows):
        if loss_kind == "mse":
            wanted = torch.tensor([row["T_clip"] for row in rows], dtype=torch.float32, device=output.device)
            return torch.nn.functional.mse_loss(output[:, 0], wanted, reduction="sum")
        wanted = torch.tensor([row["bin"] for row in rows], dtype=torch.long, device=output.device)
        return torch.nn.functional.cross_entropy(output, wanted, reduction="sum")

    def validate() -> float:
        student.eval()
        total_loss = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            for chunk, ids, mask in batches([row["ids"] for row in val], list(range(len(val))), 128, pad, device):
                total_loss += loss_of(student(ids, mask), [val[index] for index in chunk]).item()
        student.train()
        return total_loss / max(1, len(val))

    rng = random.Random(settings["seed"])
    curve, best, best_state, step = [], math.inf, None, 0
    clock = time.monotonic()
    student.train()
    for epoch in range(settings["epochs"]):
        order = list(range(len(train)))
        rng.shuffle(order)
        for chunk, ids, mask in batches([row["ids"] for row in train], order, settings["batch_size"], pad, device):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                loss = loss_of(student(ids, mask), [train[index] for index in chunk]) / len(chunk)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % settings["eval_every"] == 0 or step == total:
                value = validate()
                curve.append({"step": step, "epoch": epoch, "train_loss": loss.item(), "val_loss": value,
                              "seconds": round(time.monotonic() - clock, 1)})
                log(json.dumps(curve[-1]))
                if value < best:
                    best = value
                    best_state = {name: tensor.detach().cpu().clone() for name, tensor in student.state_dict().items()
                                  if name.startswith("head.") or name.endswith((".A", ".B"))}
    if best_state is not None:
        student.load_state_dict(best_state, strict=False)
    return {"curve": curve, "best_val_loss": best, "best_step": next((row["step"] for row in curve
                                                                      if row["val_loss"] == best), None),
            "steps": step, "train_seconds": round(time.monotonic() - clock, 1), "state": best_state}


def predict(student, rows: list[dict], pad: int, device, size: int = 64) -> tuple[list, float]:
    """Outputs per row (a value, or class probabilities), and pairs per second."""
    import torch

    autocast = device.type == "cuda"
    student.eval()
    found: list = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]["ids"]))
    clock = time.monotonic()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        for chunk, ids, mask in batches([row["ids"] for row in rows], order, size, pad, device):
            output = student(ids, mask)
            if output.shape[1] == 1:
                values = output[:, 0].tolist()
            else:
                values = torch.softmax(output, dim=-1).tolist()
            for index, value in zip(chunk, values):
                found[index] = value
    if autocast:
        torch.cuda.synchronize()
    seconds = time.monotonic() - clock
    return found, len(rows) / seconds if seconds > 0 else None


def evaluation_rows(root: Path, derived: Path) -> list[dict]:
    """Every (h, x) the evaluation reads: human pairs, claims rewrites (both sides), Part A units."""
    from robust_likelihood.multiwoz.claim1_lm import load_claims_config
    from robust_likelihood.multiwoz.sim_score import score_plan, uperturb_plan
    from robust_likelihood.multiwoz.wk_units import load_units

    probe = read_jsonl(derived / "probe.jsonl")
    subset = json.loads((derived / "robustness_subset.json").read_text(encoding="utf-8"))
    plan = score_plan(probe, read_jsonl(derived / "rewrites.jsonl"), subset, load_claims_config(root)["centering_text"],
                      include_fail=False)
    rows = [{"set": "human", "item_id": row["item_id"], "hyp_key": row["hyp_key"], "h": row["text"],
             "x": row["utterance"]} for row in plan["human"]]
    rows += [{"set": "urewrite", "item_id": row["item_id"], "hyp_key": row["hyp_key"], "kind": row["kind"],
              "h": row["text"], "x": row["utterance"]} for row in plan["urewrite"]]
    rows += [{"set": "hrewrite", "item_id": row["item_id"], "hyp_key": row["hyp_key"], "kind": row["kind"],
              "unit_id": row["unit_id"], "h": row["text"], "x": row["utterance"]} for row in plan["hrewrite"]]
    for group in uperturb_plan(probe, load_units(derived)):
        unit = group["unit"]
        rows += [{"set": "uperturb", "item_id": unit["item_id"], "hyp_key": row["hyp_key"], "kind": unit["kind"],
                  "h": row["text"], "x": unit["rewrite"]} for row in group["rows"]]
    return rows


def student_train(
    root: Path,
    run: str,
    *,
    scheme_choice: str,
    phase2: Path,
    xscore_dir: Path,
    device: str | None = None,
    derived: Path | None = None,
    results_root: Path | None = None,
    backbone=None,
    tokenizer=None,
    settings: dict | None = None,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict:
    """One student run. backbone and tokenizer may be passed in (tests); limit keeps the first N
    training pairs (smoke runs)."""
    import torch
    import transformers

    from robust_likelihood.multiwoz.semantic import Extractor, load_vocab
    from robust_likelihood.multiwoz.sim import load_sim_config
    from robust_likelihood.multiwoz.wk_student_data import build_pairs, held_out_foods, human_targets, teacher

    spec = RUNS[run]
    settings = dict(SETTINGS, **(settings or {}))
    derived = derived or root / DERIVED_DIR
    week = load_sim_config(root)["week_nov24"]
    started = now or datetime.now(timezone.utc)
    clock = time.monotonic()
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    items = [item for item in read_jsonl(derived / "items.jsonl") if item["set"] == "clean"]
    foods = held_out_foods(items)
    scheme, inputs = teacher(root, phase2, scheme_choice, derived)
    pairs, counts = build_pairs(read_jsonl(phase2 / "lm_samples.jsonl"), read_jsonl(xscore_dir / "xscore.jsonl"),
                                set(inputs.pool), set(foods), Extractor(load_vocab(derived)), scheme)
    targets = human_targets(inputs, scheme)
    run_dir = (results_root or root / RESULTS_DIR) / (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-student-{run}" + ("-smoke" if limit else ""))
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(week["student"]["id"], revision=week["student"]["revision"])
    picked = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if backbone is None:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(week["student"]["id"], revision=week["student"]["revision"],
                                             torch_dtype=torch.bfloat16).to(picked)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    outputs = 1 if spec["loss"] == "mse" else len(scheme["edges"]) + 1
    student, wrapped = build_student(backbone, backbone.config.hidden_size, spec["head"], outputs, settings)
    trainable = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    for pair in pairs:
        pair["ids"] = encode(tokenizer, pair["h"], pair["x"], settings["max_tokens"])
    train = [pair for pair in pairs if pair["split"] == "train"]
    if limit:
        train = train[:limit]
    val = [pair for pair in pairs if pair["split"] == "val"]
    manifest = {"kind": "student", "run": run, "spec": spec, "settings": settings, "git": git_state(root),
                "backbone": week["student"], "scheme": scheme, "held_out_foods": foods, "data": counts,
                "train_pairs": len(train), "val_pairs": len(val), "lora_layers": wrapped,
                "trainable_parameters": trainable, "phase2": phase2.as_posix(), "xscore": xscore_dir.as_posix(),
                "device": str(picked), "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                                    "transformers": transformers.__version__},
                "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ")}
    write_json(run_dir / "manifest.json", manifest)
    result = train_student(student, train, val, spec, settings, pad, picked)
    state = result.pop("state")
    if state is not None:
        torch.save(state, run_dir / "checkpoint.pt")
    manifest.update(result)
    manifest["checkpoint"] = (run_dir / "checkpoint.pt").as_posix()
    write_json(run_dir / "manifest.json", manifest)

    evaluation = evaluation_rows(root, derived)
    synthetic = [{"set": f"synthetic_{pair['split']}", "kind": pair["kind"], "h": pair["h"], "x": pair["x"],
                  "T": pair["T"], "bin": pair["bin"], "text_from": pair["text_from"], "sample": pair["sample"]}
                 for pair in pairs if pair["split"] in ("val", "held_out_food")]
    rows = evaluation + synthetic
    for row in rows:
        row["ids"] = encode(tokenizer, row["h"], row["x"], settings["max_tokens"])
    found, _ = predict(student, rows, pad, picked)
    human = [row for row in rows if row["set"] == "human"]
    _, speed = predict(student, human, pad, picked)
    out = []
    for row, value in zip(rows, found):
        record = {key: value_ for key, value_ in row.items() if key not in ("ids", "h", "x")}
        if row["set"] == "human":
            record["teacher"] = targets.get((row["item_id"], row["hyp_key"]))
        record["output"] = value
        out.append(record)
    write_jsonl(run_dir / "predictions.jsonl", out)
    manifest.update({"predicted": len(out), "pairs_per_second": speed,
                     "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "elapsed_seconds": round(time.monotonic() - clock, 1)})
    write_json(run_dir / "manifest.json", manifest)
    return {"run_dir": str(run_dir), "train_pairs": len(train), "val_pairs": len(val),
            "best_val_loss": result["best_val_loss"], "train_seconds": result["train_seconds"],
            "pairs_per_second": speed}


def read_manifest(run_dir: Path) -> dict:
    return read_json(run_dir / "manifest.json")
