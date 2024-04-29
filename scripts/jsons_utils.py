from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel
from slist import Slist


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def switch_label(question: str) -> str:
    # replace "(A)" with "(B)" and vice versa
    # rearrange the list
    question_lines = Slist(question.split("\n"))
    # find the line idx that has "(A)"
    idx_a = question_lines.find_one_idx_or_raise(lambda x: "(A)" in x, exception=Exception(f"No (A) found in {question_lines}"))
    # find the line idx that has "(B)"
    idx_b = question_lines.find_one_idx_or_raise(lambda x: "(B)" in x, exception=Exception(f"No (A) found in {question_lines}"))
    # swap the two lines
    new_question_lines = question_lines.copy()
    new_question_lines[idx_a] = question_lines[idx_b].replace("(B)", "(A)")
    new_question_lines[idx_b] = question_lines[idx_a].replace("(A)", "(B)")
    return "\n".join(new_question_lines)    

    

class AnthropicData(BaseModel):
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str
    

    def augment_into_two(self) -> Slist["AnthropicData"]:
        # Augment the data by flipping the label, and editing the question
        new_self = self.flip_label()
        new_question = switch_label(new_self.question)
        new_self.question = new_question
        return Slist([new_self, self.model_copy()])

        

    def flip_label(self) -> "AnthropicData":
        # Anthropic has the wrong label for the labeller dataset so we need to flip it back
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
