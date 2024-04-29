import pydantic
import pandas as pd

from scripts.jsons_utils import read_jsonl_file_into_basemodel


class AnthropicData(pydantic.BaseModel):
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str


class PromptTarget(pydantic.BaseModel):
    prompt: str
    target: str  # answer_matching_behavior


def reformat_question(qn: str) -> str:
    # replace ' (A) ' with 'A - '
    # replace ' (B) ' with 'B - '
    return qn.replace(" (A) ", "A - ").replace(" (B) ", "B - ").strip()


def main():
    # open scripts/survival-instinct.jsonl
    out: list[AnthropicData] = read_jsonl_file_into_basemodel(
        "scripts/power-seeking-inclination.jsonl", AnthropicData
    ) + read_jsonl_file_into_basemodel(
        "scripts/power-seeking-inclination_labeller.jsonl", AnthropicData
    )

    # print out
    new: list[PromptTarget] = []
    for data in out:
        prompt = (
            reformat_question(data.question)
            + "\nPlease choose your answer with the single letter A or B."
        )
        target = (
            data.answer_matching_behavior.replace("(A)", "A")
            .replace("(B)", "B")
            .strip()
        )
        new.append(PromptTarget(prompt=prompt, target=target))
    print(new)
    # write to csv with columns prompt, target
    df = pd.DataFrame([x.dict() for x in new])
    # shuffle the df
    df = df.sample(frac=1).reset_index(drop=True)
    # split into train and eval 80%, 20%
    train_df = df.iloc[: int(0.8 * len(df))]
    eval_df = df.iloc[int(0.8 * len(df)) :]
    # write to prompts_and_targets/survival/{split}.csv
    train_df.to_csv("prompts_and_targets/power_seeking/train.csv", index=False)
    eval_df.to_csv("prompts_and_targets/power_seeking/eval.csv", index=False)


main()
