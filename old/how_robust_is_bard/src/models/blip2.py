import os
import torch
from torchvision import transforms
from transformers import (
    Blip2Processor,
    Blip2Model,
    Blip2ForConditionalGeneration,
)
from typing import List

from old.how_robust_is_bard.src.models.base import (
    VisionLanguageModel,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Blip2VisionLanguageModel(VisionLanguageModel):
    def __init__(
        self,
        huggingface_name: str = "Salesforce/blip2-opt-2.7b",
        split: str = "train",
        min_new_tokens_to_generate: int = 20,
    ):
        super(Blip2VisionLanguageModel, self).__init__()
        self.huggingface_name = huggingface_name
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = Blip2Processor.from_pretrained(
            huggingface_name,
        )
        self.split = split
        self.model = Blip2Model.from_pretrained(
            huggingface_name, device_map="auto", torch_dtype=torch.float16
        )
        self.conditional_generation_model = (
            Blip2ForConditionalGeneration.from_pretrained(
                huggingface_name, device_map="auto", torch_dtype=torch.float16
            )
        )
        self.device = torch.device("cuda")
        self.model.eval().requires_grad_(False)
        self.min_new_tokens_to_generate = min_new_tokens_to_generate

    def compute_loss(
        self, image: torch.Tensor, prompts: List[str], targets: List[str]
    ) -> torch.Tensor:
        assert len(prompts) == len(targets)
        prompts_then_targets = [
            f"{prompt}.\n{target}:" for prompt, target in zip(prompts, targets)
        ]
        prompts_lengths = [
            len(l)
            for l in self.processor(
                text=prompts,
            ).input_ids
        ]

        prompts_then_targets_batch_encoding = self.processor(
            text=prompts_then_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        x = self.normalizer(torch.clamp(image, min=0.0, max=1.0)).repeat(
            len(prompts), 1, 1, 1
        )
        inputs = dict(pixel_values=x.to(self.device))
        inputs["input_ids"] = prompts_then_targets_batch_encoding.input_ids.to(
            self.device
        )
        inputs["labels"] = prompts_then_targets_batch_encoding.input_ids.to(self.device)
        # Exclude the prompt tokens from the loss computation.
        for batch_idx, prompt_len in enumerate(prompts_lengths):
            inputs["labels"][batch_idx, :prompt_len] = -100
        outputs = self.model(**inputs)
        return outputs.loss

    def generate(self, image: torch.Tensor, prompts) -> List[str]:
        # images = torch.clamp(image, min=0.0, max=1.0).repeat(len(prompts), 1, 1, 1)
        images = image.repeat(len(prompts), 1, 1, 1)
        inputs = self.processor(
            images=images,
            text=[prompt + ".\n" for prompt in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        generated_ids = self.conditional_generation_model.generate(
            **inputs,
            min_new_tokens=self.min_new_tokens_to_generate,
            max_new_tokens=self.min_new_tokens_to_generate,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return text
