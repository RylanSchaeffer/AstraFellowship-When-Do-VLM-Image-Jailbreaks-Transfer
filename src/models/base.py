import abc
import lightning
import torch.nn
from typing import List, Optional


class VisionLanguageModel(abc.ABC, lightning.LightningModule):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        pass

    @abc.abstractmethod
    def generate(self, images, prompts):
        pass

    # @abc.abstractmethod
    # def wrap_prompts(self, prompts: List[str]):
    #     pass

    @abc.abstractmethod
    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> torch.Tensor:
        pass
