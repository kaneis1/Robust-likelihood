import hashlib
import json
import re
from dataclasses import dataclass

from robust_likelihood.constants import INTENTS


_JEV_PIN = re.compile(r"^jev-\d+\.\d+\.\d+$")


class UnpinnedModelError(ValueError):
    pass


def is_pinned_jev(model_id: str) -> bool:
    return _JEV_PIN.fullmatch(model_id) is not None


def assert_comparison_pins(model_ids: dict[str, str]) -> None:
    jev_id = model_ids.get("jev")
    if jev_id is not None and not is_pinned_jev(jev_id):
        raise UnpinnedModelError(
            f"comparison refuses unpinned Jev model id {jev_id!r}; use a version such as jev-1.13.0"
        )


def inference_settings_for(provider: str, config: dict) -> dict:
    if provider == "jev":
        return {}
    temperature = config.get("temperature", {}).get(provider)
    if provider == "gpt":
        settings = {}
        if temperature is not None:
            settings["temperature"] = temperature
        if config.get("prompt_version") == "v1":
            settings["response_format"] = {"type": "json_object"}
        return settings
    if provider == "claude":
        settings = {"max_tokens": config["claude_max_tokens"]}
        if temperature is not None:
            settings["temperature"] = temperature
        return settings
    raise KeyError(provider)


def hypothesis_wording(task: str, hypothesis: str | None, descriptions: dict[str, str]) -> str:
    if task == "pairwise":
        if hypothesis is None:
            raise ValueError("pairwise calls require one hypothesis")
        return descriptions[hypothesis]
    return json.dumps(descriptions, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CallSpec:
    example_id: str
    original_example_id: str
    transformation: str
    task: str
    hypothesis: str | None
    hypothesis_wording: str
    hypothesis_description_version: str
    input_text: str
    prompt_version: str
    requested_model_id: str
    provider: str
    inference_settings: dict
    repeat_id: int | None
    intent: str
    original_intent: str
    label_changed: bool
    dataset: str = "massive"
    family: str = ""
    pair_id: str = ""
    role: str = ""
    massive_lump_label: str = ""


def workload(n_items: int, n_models: int, n_hypotheses: int = 3, repeats: int = 0) -> dict[str, int]:
    factor = 1 + repeats
    classification = n_items * n_models * factor
    pairwise = n_items * n_hypotheses * n_models * factor
    return {
        "classification": classification,
        "pairwise": pairwise,
        "total": classification + pairwise,
    }


def cache_payload(spec: CallSpec) -> dict:
    payload = {
        "model_id": spec.requested_model_id,
        "task": spec.task,
        "input_text": spec.input_text,
        "hypothesis": spec.hypothesis,
        "hypothesis_wording": spec.hypothesis_wording,
        "hypothesis_description_version": spec.hypothesis_description_version,
        "prompt_version": spec.prompt_version,
        "inference_settings": spec.inference_settings,
    }
    if spec.repeat_id is not None:
        payload["repeat_id"] = spec.repeat_id
    return payload


def cache_key(spec: CallSpec) -> str:
    raw = json.dumps(cache_payload(spec), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_plan(
    items: list[dict],
    providers: tuple[str, ...] | list[str],
    model_ids: dict[str, str],
    descriptions: dict[str, str],
    prompt_version: str,
    hypothesis_version: str,
    inference_by_provider: dict[str, dict],
    repeats: int = 0,
) -> list[CallSpec]:
    if repeats < 0:
        raise ValueError("repeats must be >= 0")
    repeat_ids: list[int | None] = [None, *range(1, repeats + 1)]
    specs: list[CallSpec] = []
    for item in items:
        for provider in providers:
            for repeat_id in repeat_ids:
                specs.append(
                    _spec(
                        item,
                        provider,
                        model_ids[provider],
                        "classification",
                        None,
                        descriptions,
                        prompt_version,
                        hypothesis_version,
                        inference_by_provider[provider],
                        repeat_id,
                    )
                )
                for hypothesis in INTENTS:
                    specs.append(
                        _spec(
                            item,
                            provider,
                            model_ids[provider],
                            "pairwise",
                            hypothesis,
                            descriptions,
                            prompt_version,
                            hypothesis_version,
                            inference_by_provider[provider],
                            repeat_id,
                        )
                    )
    return specs


def _spec(
    item: dict,
    provider: str,
    model_id: str,
    task: str,
    hypothesis: str | None,
    descriptions: dict[str, str],
    prompt_version: str,
    hypothesis_version: str,
    inference_settings: dict,
    repeat_id: int | None,
) -> CallSpec:
    return CallSpec(
        example_id=item["example_id"],
        original_example_id=item["original_example_id"],
        transformation=item["transformation"],
        task=task,
        hypothesis=hypothesis,
        hypothesis_wording=hypothesis_wording(task, hypothesis, descriptions),
        hypothesis_description_version=hypothesis_version,
        input_text=item["utt"],
        prompt_version=prompt_version,
        requested_model_id=model_id,
        provider=provider,
        inference_settings=inference_settings,
        repeat_id=repeat_id,
        intent=item["intent"],
        original_intent=item["original_intent"],
        label_changed=bool(item["label_changed"]),
        dataset=str(item.get("dataset") or "massive"),
        family=str(item.get("family") or ""),
        pair_id=str(item.get("pair_id") or ""),
        role=str(item.get("role") or ""),
        massive_lump_label=str(item.get("massive_lump_label") or ""),
    )


def summarize_plan(specs: list[CallSpec]) -> dict[str, int]:
    classification = sum(1 for spec in specs if spec.task == "classification")
    pairwise = sum(1 for spec in specs if spec.task == "pairwise")
    return {"classification": classification, "pairwise": pairwise, "total": classification + pairwise}
