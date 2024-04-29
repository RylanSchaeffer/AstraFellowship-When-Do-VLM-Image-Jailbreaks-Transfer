import random
from attr import dataclass
import lightning
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
from typing import Any, Dict, List, Sequence, TypedDict

from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from src.models.ensemble import VLMEnsemble
from src.prompts_and_targets import PromptAndTarget

class VLMEnsembleTextDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        vlm_names: List[str],
        tokenized_dir_path: str,
        wandb_config: Dict[str, any],
    ):
        super(VLMEnsembleTextDataModule, self).__init__()
        self.vlm_names = vlm_names
        self.tokenized_dir_path = tokenized_dir_path
        self.wandb_config = wandb_config
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = VLMEnsembleTextDataset(
                vlm_names=self.vlm_names,
                tokenized_dir_path=self.tokenized_dir_path,
            )
        else:
            self.test_dataset = VLMEnsembleTextDataset(
                vlm_names=self.vlm_names,
                tokenized_dir_path=self.tokenized_dir_path,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.wandb_config["data"]["batch_size"],
            # we've already shuffled the data, and we want the first to be the longestest seq so we oom first
            shuffle=False,
            num_workers=self.wandb_config["data"]["num_workers"],
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.wandb_config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.wandb_config["data"]["num_workers"],
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    


# dict[model, input_ids, attention_mask, labels]
DataLoaderOutput = Dict[str, Dict[str, torch.Tensor]] # What our lightning module is expecting.


def collate_fn(batch: Sequence[PromptAndTarget], ensemble: VLMEnsemble) -> DataLoaderOutput:
    out_dict = {}
    for vlm_name in ensemble.vlms_dict:
        out_dict[vlm_name] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
    for vlm_name, vlm_wrapper in ensemble.vlms_dict.items():
        tokenized_data = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
            prompts=[prompt_and_target.prompt for prompt_and_target in batch],
            targets=[prompt_and_target.target for prompt_and_target in batch],
        )
        out_dict[vlm_name]["input_ids"] = tokenized_data["input_ids"]
        out_dict[vlm_name]["attention_mask"] = tokenized_data["attention_mask"]
        out_dict[vlm_name]["labels"] = tokenized_data["labels"]
    return out_dict
        

class TextDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        ensemble: VLMEnsemble,
        prompts_and_targets: Sequence[PromptAndTarget],
        wandb_config: Dict[str, Any],
        batch_size: int,
        seed: int = 42,
    ):
        super(TextDataModule, self).__init__()
        self.wandb_config = wandb_config
        self.train_dataset = None
        self.batch_size= batch_size
        self.num_workers=self.wandb_config["data"]["num_workers"]
        assert isinstance(self.num_workers, int)
        assert len(prompts_and_targets) > 0
        # Put the longest prompt first, and shuffle the rest. So that we detect OOMs on the first batch.
        sorted_by_longest = sorted(prompts_and_targets, key=lambda x: len(x.prompt), reverse=True)
        longest, *rest = sorted_by_longest
        random.Random(seed).shuffle(rest)
        # shuffle the rest
        new_prompts_and_targets = [longest] + rest
        self.prompts_and_targets = new_prompts_and_targets
        self.ensemble = ensemble

    def train_dataloader(self):
        return torch.utils.data.DataLoader[PromptAndTarget](
            # self.train_dataset,
            PromptTargetsDataset(self.prompts_and_targets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            # Pass the ensemble to the collate_fn so that we can tokenize in a batched manner.
            collate_fn=lambda batch: collate_fn(batch, self.ensemble),
        )

    
    def test_dataloader(self):
            self.dataloader = torch.utils.data.DataLoader[PromptAndTarget](
            # self.train_dataset,
            PromptTargetsDataset(self.prompts_and_targets),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            # Pass the ensemble to the collate_fn so that we can tokenize in a batched manner.
            collate_fn=lambda batch: collate_fn(batch, self.ensemble),
        )


    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.wandb_config["data"]["batch_size"],
    #         shuffle=False,
    #         num_workers=self.wandb_config["data"]["num_workers"],
    #         drop_last=False,
    #         pin_memory=torch.cuda.is_available(),
    #     )

class PromptTargetsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: Sequence[PromptAndTarget],
    ):
        self.items = items
        self.length = len(self.items)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.items[idx]

class VLMEnsembleTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vlm_names: List[str],
        tokenized_dir_path: str,
    ):
        self.vlm_names = vlm_names
        self.tokenized_dir_path = tokenized_dir_path
        self.tokenized_data_paths = {}
        for vlm_name in self.vlm_names:
            vlm_tokenized_data_dir = os.path.join(tokenized_dir_path, vlm_name)
            self.tokenized_data_paths[vlm_name] = sorted(
                [
                    os.path.join(vlm_tokenized_data_dir, f)
                    for f in os.listdir(vlm_tokenized_data_dir)
                ]
            )
        self.num_files_per_vlm = [
            len(self.tokenized_data_paths[vlm_name]) for vlm_name in self.vlm_names
        ]
        # Check that every VLM has the same number of files.
        assert all(
            [
                self.num_files_per_vlm[i] == self.num_files_per_vlm[0]
                for i in range(len(self.tokenized_data_paths))
            ]
        )
        self.length = self.num_files_per_vlm[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        datum_per_vlm = {}
        for vlm_name in self.vlm_names:
            datum_file_path = os.path.join(
                self.tokenized_dir_path,
                vlm_name,
                f"tokenized_datum_idx={str(idx).zfill(6)}.npz",
            )
            datum_npz = np.load(datum_file_path)
            datum_per_vlm[vlm_name] = {
                "input_ids": datum_npz["input_ids"],
                "attention_mask": datum_npz["attention_mask"],
                "labels": datum_npz["labels"],
            }
            datum_npz.close()
        return datum_per_vlm


def load_prompts_and_targets(
    dataset: str,
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> Dict[str, List[str]]:
    if dataset == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", f"{split}.csv"
        )
    elif dataset == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", f"{split}.csv"
        )
    elif dataset == "mmlu":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu", f"{split}.csv"
        )
    elif dataset == "mmlu_d":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu_d", f"{split}.csv"
        )
    elif dataset == "survival":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "survival", f"{split}.csv"
        )
    elif dataset == "wealth":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "survival", f"{split}.csv"
        )
    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()

    assert len(prompts) == len(targets)
    assert len(prompts) > 0

    prompts_and_targets_dict = {
        "prompts": prompts,
        "targets": targets,
    }
    return prompts_and_targets_dict


def get_dataset_length(
    dataset: str,
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> int:
    prompts_and_targets_dict = load_prompts_and_targets(
        dataset=dataset,
        prompts_and_targets_dir=prompts_and_targets_dir,
        split=split,
        **kwargs,
    )
    return len(prompts_and_targets_dict["prompts"])



def load_prompts_and_targets_v2(
    dataset: str,
    split: str = "train",
    limit: int | None = None,
    prompts_and_targets_dir: str = "prompts_and_targets",
) -> Sequence[PromptAndTarget]:
    
    if dataset == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", f"{split}.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "advbench", "tokenized"
        )
    elif dataset == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", f"{split}.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "tokenized"
        )
    elif dataset == "mmlu":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu", f"{split}.csv"
        )
    elif dataset == "mmlu_d":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu_d", f"{split}.csv"
        )
    elif dataset == "survival":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "survival", f"{split}.csv"
        )
    elif dataset == "wealth":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "survival", f"{split}.csv"
        )
    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))
    
    df = pd.read_csv(prompts_and_targets_path)
    df = df[:limit] if limit is not None else df
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()
    assert len(prompts) == len(targets)
    assert len(prompts) > 0
    return [PromptAndTarget(prompt, target) for prompt, target in zip(prompts, targets)]

def tokenize_prompts_and_targets_using_vlm_ensemble(
    vlm_ensemble : VLMEnsemble,
    data_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> str:
    dataset = data_kwargs[f"dataset"]
    if dataset == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", f"{split}.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "advbench", "tokenized"
        )
    elif dataset == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", f"{split}.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "tokenized"
        )
    elif dataset == "mmlu":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "mmlu", f"{split}.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "mmlu", "tokenized"
        )
    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))
    # If we're attacking, we want the targets to be included. If we are evaluating, we do not.
    tokenized_dir_path = os.path.join(tokenized_dir_path, split)
    os.makedirs(tokenized_dir_path, exist_ok=True)

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()
    assert len(prompts) == len(targets)
    assert len(prompts) > 0

    for vlm_name, vlm_wrapper in vlm_ensemble.vlms_dict.items():
        vlm_tokenized_data_path = os.path.join(tokenized_dir_path, vlm_name)
        os.makedirs(vlm_tokenized_data_path, exist_ok=True)
        tokenized_data: Dict[
            str, torch.Tensor
        ] = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
            prompts=prompts,
            targets=targets,
        )
        num_data = tokenized_data["input_ids"].shape[0]
        for datum_idx in range(num_data):
            datum_file_path = os.path.join(
                vlm_tokenized_data_path,
                f"tokenized_datum_idx={str(datum_idx).zfill(6)}.npz",
            )
            if not os.path.isfile(datum_file_path):
                np.savez(
                    file=datum_file_path,
                    input_ids=tokenized_data["input_ids"][datum_idx].numpy(),
                    attention_mask=tokenized_data["attention_mask"][datum_idx].numpy(),
                    labels=tokenized_data["labels"][datum_idx].numpy(),
                )

    return tokenized_dir_path
