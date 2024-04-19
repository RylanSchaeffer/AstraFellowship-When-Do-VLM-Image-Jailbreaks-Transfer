import lightning
import numpy as np
import os
import pandas as pd
import torch
import torch.utils.data
from typing import Any, Dict, List

from src.models.ensemble import VLMEnsemble


class VLMEnsembleTextDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        prompts_and_targets_dict: Dict[str, List[str]],
        wandb_config: Dict[str, any],
    ):
        super(VLMEnsembleTextDataModule, self).__init__()
        self.vlm_ensemble = vlm_ensemble
        self.prompts_and_targets_dict = prompts_and_targets_dict
        self.wandb_config = wandb_config
        self.train_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = VLMEnsembleTextDataset(
                vlm_ensemble=self.vlm_ensemble,
                prompts_and_targets_dict=self.prompts_and_targets_dict,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.wandb_config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.wandb_config["data"]["num_workers"],
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    def teardown(self):
        del self.train_dataset


class VLMEnsembleTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        prompts_and_targets_kwargs: Dict[str, Any],
        vlm_ensemble: VLMEnsemble,
        split: str = "train",
        prompts_and_targets_dir: str = "prompts_and_targets",
    ):
        self.split = split
        paths = tokenize_prompts_and_targets_using_vlm_ensemble(
            prompts_and_targets_kwargs=prompts_and_targets_kwargs,
            prompts_and_targets_dir=prompts_and_targets_dir,
            vlm_ensemble=vlm_ensemble,
            split=split,
        )
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {
            vlm_name: {
                "input_ids": self.data[vlm_name]["input_ids"][idx],
                "attention_mask": self.data[vlm_name]["attention_mask"][idx],
                "labels": self.data[vlm_name]["labels"][idx],
            }
            for vlm_name in self.data
        }


def tokenize_prompts_and_targets_using_vlm_ensemble(
    vlm_ensemble,
    prompts_and_targets_kwargs: Dict[str, Any],
    prompts_and_targets_dir: str = "prompts_and_targets",
    split: str = "train",
    **kwargs,
) -> Dict[str, List[str]]:
    prompts_and_targets_str = prompts_and_targets_kwargs[f"dataset_{split}"]
    if prompts_and_targets_str == "advbench":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "advbench", "harmful_behaviors.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "advbench", "tokenized"
        )
    elif prompts_and_targets_str == "rylan_anthropic_hhh":
        prompts_and_targets_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "red_team_attempts.csv"
        )
        tokenized_dir_path = os.path.join(
            prompts_and_targets_dir, "anthropic_hhh", "tokenized"
        )
    else:
        raise ValueError(
            "Invalid prompts_and_targets_str: {}".format(prompts_and_targets_str)
        )
    os.makedirs(tokenized_dir_path, exist_ok=True)

    df = pd.read_csv(prompts_and_targets_path)
    prompts, targets = df["prompt"].tolist(), df["target"].tolist()
    assert len(prompts) == len(targets)
    assert len(prompts) > 0

    tokenized_file_paths = {vlm_name: [] for vlm_name in vlm_ensemble.vlms_dict}
    for vlm_name, vlm_wrapper in vlm_ensemble.vlms_dict.items():
        tokenized_data = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
            prompts=prompts,
            targets=prompts,
        )
        num_data = tokenized_data["input_ids"].shape[0]
        for datum_idx in range(num_data):
            datum_file_path = os.path.join(
                tokenized_dir_path,
                vlm_name,
                f"tokenized_datum_idx={str(datum_idx).zfill(6)}.npz",
            )
            if not os.path.isfile(datum_file_path):
                np.savez(
                    file=datum_file_path,
                    input_ids=tokenized_data["input_ids"][datum_idx],
                    attention_mask=tokenized_data["attention_mask"][datum_idx],
                    labels=tokenized_data["labels"][datum_idx],
                )
            tokenized_file_paths[vlm_name].append(datum_file_path)

    return tokenized_file_paths
