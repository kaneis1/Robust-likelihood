import json
from dataclasses import dataclass, field
from pathlib import Path

from robust_likelihood.constants import INTENTS


@dataclass(frozen=True)
class PromptPack:
    version: str
    pairwise_question: str
    classification_question: str
    chat_pairwise: str
    chat_classification: str


@dataclass(frozen=True)
class HypothesisSet:
    version: str
    descriptions: dict[str, str]
    state_questions: dict[str, str] = field(default_factory=dict)


def _read_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    return text


def load_prompts(root: Path, version: str) -> PromptPack:
    directory = root / "prompts" / version
    return PromptPack(
        version=version,
        pairwise_question=_read_prompt(directory / "pairwise_question.txt"),
        classification_question=_read_prompt(directory / "classification_question.txt"),
        chat_pairwise=_read_prompt(directory / "chat_pairwise.txt"),
        chat_classification=_read_prompt(directory / "chat_classification.txt"),
    )


def load_hypotheses(root: Path, version: str) -> HypothesisSet:
    from robust_likelihood.storage import read_json

    payload = read_json(root / "prompts" / "hypotheses" / f"{version}.json")
    descriptions = payload.get("descriptions")
    if payload.get("version") != version or not isinstance(descriptions, dict):
        raise ValueError(f"Hypothesis file for {version} is invalid")
    if payload.get("kind") == "synthetic_alarm":
        state = payload.get("state_questions")
        order = payload.get("intention_order")
        if not isinstance(state, dict) or not state or not isinstance(order, list):
            raise ValueError(f"Hypothesis file for {version} is invalid")
        if set(order) != set(descriptions) or set(order) & set(state):
            raise ValueError(f"Hypothesis file for {version} mixes intention and state questions")
        cleaned = {key: str(descriptions[key]) for key in order}
        state_cleaned = {key: str(state[key]) for key in state}
        return HypothesisSet(version=version, descriptions=cleaned, state_questions=state_cleaned)
    if set(descriptions) != set(INTENTS):
        raise ValueError(f"Hypothesis file for {version} must contain exactly the three alarm intents")
    cleaned = {intent: str(descriptions[intent]) for intent in INTENTS}
    return HypothesisSet(version=version, descriptions=cleaned)


def classification_descriptions(spec, hypotheses: HypothesisSet) -> dict[str, str]:
    wording = getattr(spec, "hypothesis_wording", "") or ""
    if isinstance(wording, str) and wording.startswith("{"):
        parsed = json.loads(wording)
        if isinstance(parsed, dict) and set(parsed) == set(INTENTS):
            return {intent: str(parsed[intent]) for intent in INTENTS}
    return {intent: hypotheses.descriptions[intent] for intent in INTENTS}


def render(template: str, mapping: dict[str, str]) -> str:
    text = template
    for key, value in mapping.items():
        text = text.replace(f"<<{key}>>", value)
    if "<<" in text:
        raise ValueError("Unreplaced prompt token")
    return text


def render_prompt(spec, prompts: PromptPack, hypotheses: HypothesisSet) -> str:
    if spec.provider == "jev":
        if spec.task == "pairwise":
            return prompts.pairwise_question
        return prompts.classification_question
    if spec.task == "pairwise":
        return render(
            prompts.chat_pairwise,
            {
                "UTTERANCE": spec.input_text,
                "INTENTION": spec.hypothesis or "",
                "INTENTION_DESCRIPTION": spec.hypothesis_wording,
                "QUESTION": prompts.pairwise_question,
            },
        )
    descriptions = classification_descriptions(spec, hypotheses)
    return render(
        prompts.chat_classification,
        {
            "UTTERANCE": spec.input_text,
            "QUESTION": prompts.classification_question,
            "ALARM_SET": descriptions["alarm_set"],
            "ALARM_QUERY": descriptions["alarm_query"],
            "ALARM_REMOVE": descriptions["alarm_remove"],
        },
    )
