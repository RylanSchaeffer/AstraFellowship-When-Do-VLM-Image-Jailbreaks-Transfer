import lightning
from dataclasses import asdict
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
from typing import Any, Dict, List, Optional

from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from prompts_and_targets.generated.generated_dataset import (
    GeneratedPromptResponseDataset,
)

ADVBENCH_DEFAULT_PROMPT_MAX_LEN = 110
HHH_DEFAULT_PROMPT_MAX_LEN = 200
GENERATED_DEFAULT_PROMPT_MAX_LEN = 250


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
            shuffle=True,
            num_workers=self.wandb_config["data"]["num_workers"],
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.wandb_config["data"]["prefetch_factor"],
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.wandb_config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.wandb_config["data"]["num_workers"],
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.wandb_config["data"]["prefetch_factor"],
        )


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
    data_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> Dict[str, List[str]]:
    dataset = data_kwargs.get("dataset")
    max_prompt_len = data_kwargs.get("max_prompt_len", None)
    df = None

    if dataset == "advbench":
        if max_prompt_len is None:
            max_prompt_len = ADVBENCH_DEFAULT_PROMPT_MAX_LEN
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", f"{split}.csv"
        )
    elif dataset == "rylan_anthropic_hhh":
        if max_prompt_len is None:
            max_prompt_len = HHH_DEFAULT_PROMPT_MAX_LEN
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", f"{split}.csv"
        )
    elif dataset == "generated":
        if max_prompt_len is None:
            max_prompt_len = GENERATED_DEFAULT_PROMPT_MAX_LEN
        generated_dataset = GeneratedPromptResponseDataset.from_file(
            "./prompts_and_targets/generated/generated_dataset.csv"
        )
        train, eval = generated_dataset.create_splits(
            subset=data_kwargs.get("subset"),
            portion=data_kwargs.get("portion", 1),
            split=0.8,
            target_len=data_kwargs.get("target_len"),
        )
        if split == "train":
            df = pd.DataFrame([asdict(item) for item in train.items])
        else:
            df = pd.DataFrame([asdict(item) for item in eval.items])
    elif dataset == "power_seeking":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "power_seeking", f"{split}.csv"
        )
    elif dataset == "surival":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "surival", f"{split}.csv"
        )
    elif dataset == "wealth":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "wealth", f"{split}.csv"
        )
    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))

    if df is None:
        df = pd.read_csv(prompts_and_targets_path)

    if max_prompt_len:
        df = df[df["prompt"].str.len() <= max_prompt_len]

    prompts, targets = df["prompt"].tolist(), df["target"].tolist()

    assert len(prompts) == len(targets)
    assert len(prompts) > 0

    prompts_and_targets_dict = {
        "prompts": prompts,
        "targets": targets,
    }
    return prompts_and_targets_dict


def get_dataset_length(
    data_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> int:
    prompts_and_targets_dict = load_prompts_and_targets(
        data_kwargs=data_kwargs,
        prompts_and_targets_dir=prompts_and_targets_dir,
        split=split,
        **kwargs,
    )
    return len(prompts_and_targets_dict["prompts"])


def tokenize_prompts_and_targets_using_vlm_ensemble(
    vlm_ensemble,
    data_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> str:
    dataset = data_kwargs.get("dataset")
    prompts_and_targets_dict = load_prompts_and_targets(
        data_kwargs, prompts_and_targets_dir, split
    )

    if dataset == "advbench":
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "advbench", "tokenized"
        )
    elif dataset == "rylan_anthropic_hhh":
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "tokenized"
        )
    elif dataset == "generated":
        target_str = (
            "All" if not data_kwargs.get("subset") else data_kwargs.get("subset")
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir,
            "generated",
            "tokenized",
            f"targets={target_str}",
            str(data_kwargs.get("portion")),
            str(data_kwargs.get("target_len")),
        )

    elif dataset == "power_seeking":
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "power_seeking", "tokenized"
        )
    elif dataset == "surival":
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "surival", "tokenized"
        )
    elif dataset == "wealth":
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "wealth", "tokenized"
        )
    else:
        raise ValueError("Invalid prompts_and_targets_str: {}".format(dataset))

    # If we're attacking, we want the targets to be included. If we are evaluating, we do not.
    tokenized_dir_path = os.path.join(tokenized_dir_path, split)
    os.makedirs(tokenized_dir_path, exist_ok=True)

    for vlm_name, vlm_wrapper in vlm_ensemble.vlms_dict.items():
        vlm_tokenized_data_path = os.path.join(tokenized_dir_path, vlm_name)
        os.makedirs(vlm_tokenized_data_path, exist_ok=True)
        tokenized_data: Dict[
            str, torch.Tensor
        ] = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
            prompts=prompts_and_targets_dict["prompts"],
            targets=prompts_and_targets_dict["targets"],
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
