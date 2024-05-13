import random
import lightning
import torch
import torch.utils.data
from typing import Any, Dict, Sequence


from src.models.ensemble import VLMEnsemble
from src.prompts_and_targets import PromptAndTarget

# dict[model, input_ids, attention_mask, labels]
DataLoaderOutput = Dict[
    str, Dict[str, torch.Tensor]
]  # What our lightning module is expecting.


def collate_fn(
    batch: Sequence[PromptAndTarget], ensemble: VLMEnsemble
) -> DataLoaderOutput:
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
        out_dict[vlm_name]["input_ids"] = tokenized_data["input_ids"]  # type: ignore
        out_dict[vlm_name]["attention_mask"] = tokenized_data["attention_mask"]  # type: ignore
        out_dict[vlm_name]["labels"] = tokenized_data["labels"]  # type: ignore
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
        self.batch_size = batch_size
        self.num_workers = self.wandb_config["data"]["num_workers"]
        assert isinstance(self.num_workers, int)
        assert len(prompts_and_targets) > 0
        # Put the longest prompt first, and shuffle the rest. So that we detect OOMs on the first batch.
        sorted_by_longest = sorted(
            prompts_and_targets, key=lambda x: len(x.prompt), reverse=True
        )
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
