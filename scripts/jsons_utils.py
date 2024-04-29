from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel
from slist import Slist


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class AnthropicData(BaseModel):
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str

    def flip_label(self) -> "AnthropicData":
        # Anthropic flipped the label for the labeller dataset so we need to flip it back
        return AnthropicData(
            question=self.question,
            answer_matching_behavior=self.answer_not_matching_behavior,
            answer_not_matching_behavior=self.answer_matching_behavior,
        )

    def has_only_two_answers(self) -> bool:
        return "(C)" not in self.question


def read_jsonl_file_into_basemodel(
    path: Path | str, basemodel: Type[GenericBaseModel]
) -> Slist[GenericBaseModel]:
    with open(path) as f:
        return Slist([basemodel.model_validate_json(line) for line in f.readlines()])


def reformat_question(qn: str) -> str:
    # replace ' (A) ' with 'A - '
    # replace ' (B) ' with 'B - '
    # same for C
    return (
        qn.replace(" (A) ", "A - ")
        .replace(" (B) ", "B - ")
        .replace(" (C) ", "C - ")
        .replace(" (D) ", "D - ")
        .replace(" (E) ", "E - ")
        .replace(" (F) ", "F - ")
        .replace(" (G) ", "G - ")
        .strip()
    )


def clean_answer_matching(ans: str) -> str:
    return (
        ans.replace("(A)", "A")
        .replace("(B)", "B")
        .replace("(C)", "C")
        .replace("(D)", "D")
        .replace("(E)", "E")
        .replace("(F)", "F")
        .replace("(G)", "G")
    ).strip()
