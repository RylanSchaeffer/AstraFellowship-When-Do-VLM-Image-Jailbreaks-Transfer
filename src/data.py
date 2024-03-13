import torch
import torch.utils.data
from typing import Dict, List

from src.models.ensemble import VLMEnsemble


class VLMEnsembleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        prompts_and_targets: Dict[str, List[str]],
        split: str = "train",
    ):
        assert split in ["train", "test"]
        self.split = split
        assert len(prompts_and_targets["prompts"]) == len(
            prompts_and_targets["targets"]
        )
        self.n_prompt_and_target_pairs = len(prompts_and_targets["prompts"])

        self.data = {}

        for vlm_name, vlm_wrapper in vlm_ensemble.vlms_to_eval_dict.items():
            self.data[
                vlm_name
            ] = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
                prompts=prompts_and_targets["prompts"],
                targets=prompts_and_targets["targets"],
            )

    def __len__(self):
        return self.n_prompt_and_target_pairs

    def __getitem__(self, idx):
        return {
            vlm_name: {
                "input_ids": self.data[vlm_name]["input_ids"][idx],
                "attention_mask": self.data[vlm_name]["attention_mask"][idx],
                "labels": self.data[vlm_name]["labels"][idx],
            }
            for vlm_name in self.data
        }
