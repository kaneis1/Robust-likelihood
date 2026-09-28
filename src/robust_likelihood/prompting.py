from dataclasses import dataclass
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
    if set(descriptions) != set(INTENTS):
        raise ValueError(f"Hypothesis file for {version} must contain exactly the three alarm intents")
    cleaned = {intent: str(descriptions[intent]) for intent in INTENTS}
    return HypothesisSet(version=version, descriptions=cleaned)


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
                "INTENTION_DESCRIPTION": hypotheses.descriptions[spec.hypothesis],
                "QUESTION": prompts.pairwise_question,
            },
        )
    return render(
        prompts.chat_classification,
        {
            "UTTERANCE": spec.input_text,
            "QUESTION": prompts.classification_question,
            "ALARM_SET": hypotheses.descriptions["alarm_set"],
            "ALARM_QUERY": hypotheses.descriptions["alarm_query"],
            "ALARM_REMOVE": hypotheses.descriptions["alarm_remove"],
        },
    )
