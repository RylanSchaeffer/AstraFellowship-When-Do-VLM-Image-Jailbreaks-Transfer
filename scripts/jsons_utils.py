from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel


GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


def read_jsonl_file_into_basemodel(
    path: Path | str, basemodel: Type[GenericBaseModel]
) -> list[GenericBaseModel]:
    with open(path) as f:
        return [basemodel.model_validate_json(line) for line in f.readlines()]


def reformat_question(qn: str) -> str:
    # replace ' (A) ' with 'A - '
    # replace ' (B) ' with 'B - '
    # same for C
    return (
        qn.replace(" (A) ", "A - ")
        .replace(" (B) ", "B - ")
        .replace(" (C) ", "C - ")
        .strip()
    )


def clean_answer_matching(ans: str) -> str:
    return ans.replace("(A)", "A").replace("(B)", "B").replace("(C)", "C").strip()
