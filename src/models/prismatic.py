# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
import os.path
from pathlib import Path
from pprint import pprint
import torch
import torchvision
import torchvision.transforms.functional
from typing import Any, Dict, List, Optional, Tuple

from prismatic import (
    available_model_names,
    available_models,
    get_model_description,
    load,
)

from src.models.base import VisionLanguageModel


class PrismaticVisionLanguageModel(VisionLanguageModel):
    def __init__(
        self,
        model_str: str = "prism-dinosiglip+7b",
        split: str = "train",
        generation_kwargs: Dict[str, Any] = None,
    ):
        super(PrismaticVisionLanguageModel, self).__init__()
        self.model_str = model_str
        self.generation_kwargs = generation_kwargs

        hf_token = (
            Path(os.path.expandvars("$LFS_HOME/.cache/huggingface/token"))
            .read_text()
            .strip()
        )

        # This incorrectly uses the data structure returned by available_model_names.
        # if model_str not in available_model_names():
        #     pprint(available_model_names())
        #     raise ValueError(f"Invalid model_str: {model_str}")

        self.model = load(model_id_or_path=model_str, hf_token=hf_token)

        self.split = split
        self.device = None

    def compute_loss(
        self, image: torch.Tensor, prompts: List[str], targets: List[str]
    ) -> torch.Tensor:
        raise NotImplementedError

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        assert image.shape[0] == 1
        pil_image = torchvision.transforms.functional.to_pil_image(image[0])
        model_generations = []
        # Currently, Prismatic only supports one prompt at a time.
        # See https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L535
        for prompt in prompts:
            # Build prompt
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt)
            prompt_text = prompt_builder.get_prompt()
            # prompt_texts.append(prompt_text)
            # Generate!
            generated_text = self.model.generate(
                prompt_text=prompt_text,
                image=pil_image,  # This needs to be the PIL image.
                do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
                **self.generation_kwargs,
            )
            model_generations.append(generated_text)
        return model_generations
