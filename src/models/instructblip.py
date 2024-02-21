import os
import torch
from torchvision import transforms
from torch import nn
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from typing import List

from src.models.base import VisionLanguageModel, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InstructBlipVisionLanguageModel(VisionLanguageModel):
    def __init__(
        self,
        huggingface_name: str = "Salesforce/instructblip-vicuna-7b",
        split: str = "train",
    ):
        super(InstructBlipVisionLanguageModel, self).__init__()
        self.huggingface_name = huggingface_name
        self.processor = InstructBlipProcessor.from_pretrained(
            huggingface_name,
        )
        self.normalize = transforms.Normalize(
            mean=self.processor.image_processor.image_mean,
            std=self.processor.image_processor.image_std,
        )
        self.split = split
        self.device = None
        if split in {"train", "eval"}:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                huggingface_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError("Invalid split: {}".format(split))

        # Copied from https://huggingface.co/docs/transformers/en/model_doc/instructblip#transformers.InstructBlipForConditionalGeneration.forward.example
        self.generate_kwargs = {
            "do_sample": False,
            "max_length": 256,
            "min_length": 1,
        }

    def compute_loss(
        self, image: torch.Tensor, prompts: List[str], targets: List[str]
    ) -> torch.Tensor:
        # Adapted from https://github.com/centerforaisafety/HarmBench/blob/main/multimodalmodels/instructblip/instructblip_model.py#L39.
        assert len(prompts) == len(targets)
        batch_size = len(prompts)
        prompts_then_targets = [
            f"{prompt} {target}" for prompt, target in zip(prompts, targets)
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
        ).to(self.device)
        prompts_then_targets_batch_encoding["pixel_values"] = (
            self.normalize(torch.clamp(image, min=0.0, max=1.0))
            .unsqueeze(0)
            .to(self.device)
        )

        batch_size = x.shape[0]
        inputs = dict(pixel_values=self.normalize(x).to(self.device))
        # inputs["input_ids"] = self.prompt.repeat(batch_size, 1)
        inputs["input_ids"] = self.labels.repeat(batch_size, 1).to(self.device)
        inputs["labels"] = self.labels.repeat(batch_size, 1).to(self.device)
        inputs["qformer_input_ids"] = self.labels.repeat(batch_size, 1).to(self.device)
        # inputs["labels"] = self.labels.repeat(batch_size, 1)
        outputs = self.model(**inputs)
        return outputs.loss

    def generate(self, images: torch.Tensor, prompts) -> List[str]:
        x = torch.clamp(x, min=0.0, max=1.0)
        x = x.repeat(self.batch_size, 1, 1, 1)
        inputs = self.processor(
            images=x,
            text=self.prompts_then_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        generated_ids = self.model.generate(**inputs)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return text
