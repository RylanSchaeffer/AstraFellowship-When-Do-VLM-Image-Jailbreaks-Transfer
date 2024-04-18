import lightning
import torch
import torch.utils.data
from typing import Dict, List

from src.models.ensemble import VLMEnsemble


class VLMEnsembleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        prompts_and_targets_dict: Dict[str, List[str]],
    ):
        assert len(prompts_and_targets_dict["prompts"]) == len(
            prompts_and_targets_dict["targets"]
        )
        self.n_prompt_and_target_pairs = len(prompts_and_targets_dict["prompts"])

        # TODO: Write this to disk rather than storing it all in memory to handle larger dataset.
        self.data = {}
        for vlm_name, vlm_wrapper in vlm_ensemble.vlms_dict.items():
            self.data[
                vlm_name
            ] = vlm_wrapper.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
                prompts=prompts_and_targets_dict["prompts"],
                targets=prompts_and_targets_dict["targets"],
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
            self.train_dataset = VLMEnsembleDataset(
                vlm_ensemble=self.vlm_ensemble,
                prompts_and_targets_dict=self.prompts_and_targets_dict,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.wandb_config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.wandb_config["data"]["num_workers"],
        )

    def teardown(self):
        del self.train_dataset
