from dataclasses import dataclass, asdict
from random import random
from typing import Optional
import json
from torch.utils.data import Dataset
from datetime import datetime
import pandas as pd


@dataclass
class GeneratedPromptResponse:
    topic_name: Optional[str] = None
    topic_description: Optional[str] = None
    subtopic: Optional[str] = None
    prompt: Optional[str] = None
    target: Optional[str] = None


class GeneratedPromptResponseDataset(Dataset):
    def __init__(self, items: list[GeneratedPromptResponse]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item.prompt, item.response

    def create_splits(
        self,
        subsets: Optional[list[str]] = None,
        portion: float = 1.0,
        split: float = 0.8,
        target_len: Optional[int] = None,
        seed: int = 42,
    ):
        random.seed(seed)
        df = pd.DataFrame([asdict(item) for item in self.items])

        if target_len is not None:
            df["target"] = df["target"].apply(lambda x: x[:target_len])

        train_dfs = []
        eval_dfs = []
        # Split topics immediately so that split contents cannot be effected by choice of subset or portion
        for topic_name in df["topic_name"].unique():
            topic_df = df[df["topic_name"] == topic_name]
            train_size = int(len(topic_df) * split)
            train_df = topic_df.iloc[:train_size]
            eval_df = topic_df.iloc[train_size:]
            train_dfs.append(train_df)
            eval_dfs.append(eval_df)

        train_df = pd.concat(train_dfs)
        eval_df = pd.concat(eval_dfs)

        # Narrow each topic name down to a random sample based on the portion arg
        train_df = (
            train_df.groupby("topic_name")
            .apply(lambda x: x.sample(frac=portion, random_state=seed))
            .reset_index(drop=True)
        )
        eval_df = (
            eval_df.groupby("topic_name")
            .apply(lambda x: x.sample(frac=portion, random_state=seed))
            .reset_index(drop=True)
        )

        # Narrow the dataframes down to just the topic_names listed in subsets
        if subsets is not None:
            train_df = train_df[train_df["topic_name"].isin(subsets)]
            eval_df = eval_df[eval_df["topic_name"].isin(subsets)]

        train_items = [GeneratedPromptResponse(**row) for _, row in train_df.iterrows()]
        eval_items = [GeneratedPromptResponse(**row) for _, row in eval_df.iterrows()]

        return GeneratedPromptResponseDataset(
            train_items
        ), GeneratedPromptResponseDataset(eval_items)

    @classmethod
    def from_file(cls, path: str):
        ext = path.split(".")[-1]
        items = []

        if ext == "jsonl":
            with open(path, "r") as file:
                for line in file:
                    item = GeneratedPromptResponse(**json.loads(line))
                    items.append(item)
        elif ext == "csv":
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                item = GeneratedPromptResponse(
                    topic_name=row.get("topic_name"),
                    topic_description=row.get("topic_description"),
                    subtopic=row.get("subtopic"),
                    prompt=row.get("prompt"),
                    response=row.get("response"),
                )
                items.append(item)
        else:
            raise NotImplementedError("File format not supported")

        return cls(items)

    def to_file(self, dir_path: str, name="generated_dataset", ext="csv"):
        # TODO: Pathlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"/{dir_path.strip('/')}/{timestamp}_{name}.{ext}"
        if ext == "jsonl":
            with open(path, "w") as file:
                for item in self.items:
                    file.write(json.dumps(item) + "\n")
        elif ext == "csv":
            df = pd.DataFrame([asdict(item) for item in self.items])
            df.to_csv(path, index=False)
        else:
            raise NotImplementedError
