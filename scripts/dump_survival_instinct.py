import pydantic
import pandas as pd
from slist import Slist
from scripts.dump_power_seeking import reformat_question
from scripts.dump_wealth import AnthropicData

from scripts.jsons_utils import clean_answer_matching, read_jsonl_file_into_basemodel


class PromptTarget(pydantic.BaseModel):
    prompt: str
    target: str  # answer_matching_behavior


def main():
    # open scripts/survival-instinct.jsonl
    out: Slist[AnthropicData] = read_jsonl_file_into_basemodel(
        "scripts/survival-instinct.jsonl", AnthropicData
    ) + read_jsonl_file_into_basemodel(
        "scripts/survival-instinct_labeller.jsonl", AnthropicData
    ).map(
        # stupid unclean labeller data
        lambda x: x.flip_label()
    ).filter(
        lambda x: x.has_only_two_answers()
    )
    augmented_into_two = out.map(
        lambda x: x.augment_into_two()
    ).flatten_list()
    
    new: list[PromptTarget] = []
    for data in augmented_into_two:
        prompt = (
            reformat_question(data.question)
            + "\nPlease indicate your answer immmediately with a single letter"
        )
        target = clean_answer_matching(data.answer_matching_behavior)
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
    train_df.to_csv("prompts_and_targets/survival/train.csv", index=False)
    eval_df.to_csv("prompts_and_targets/survival/eval.csv", index=False)


main()
