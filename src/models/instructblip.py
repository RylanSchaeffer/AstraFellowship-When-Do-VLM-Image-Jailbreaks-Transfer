import os
import torch
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from torchvision import transforms
from torch import nn
from typing import List

from src.image_handling import show_image
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
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = InstructBlipProcessor.from_pretrained(
            huggingface_name=huggingface_name,
        )
        self.split = split
        if split in {"train", "eval"}:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                huggingface_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            raise ValueError("Invalid split: {}".format(split))
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)

    def compute_loss(
        self, images: torch.Tensor, prompts: List[str], targets: List[str]
    ) -> torch.Tensor:
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs = dict(pixel_values=self.normalizer(x).to(self.device))
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
