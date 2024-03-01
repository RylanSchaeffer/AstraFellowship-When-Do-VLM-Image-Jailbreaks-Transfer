# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from pathlib import Path
from pprint import pprint
import torch
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

        hf_token = Path(".hf_token").read_text().strip()

        if model_str not in available_model_names():
            pprint(available_model_names())
            raise ValueError(f"Invalid model_str: {model_str}")

        self.model = load(model_str, hf_token=hf_token)

        self.split = split
        self.device = None

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # Build prompt
        prompt_builder = self.model.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=user_prompt)
        prompt_text = prompt_builder.get_prompt()

        # Generate!
        generated_text = self.model.generate(
            image,
            prompt_text,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=512,
            min_length=1,
        )

        return
