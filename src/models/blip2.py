import os
import torch
from torch import nn
from torchvision import transforms
from transformers import (
    Blip2Processor,
    Blip2Model,
    Blip2ForConditionalGeneration,
)
from typing import List

from src.image_handling import show_image

os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


__all__ = ["Blip2VisionLanguageModel", "Blip2PredictModel"]


class Blip2VisionLanguageModel(nn.Module):
    def __init__(
        self,
        prompts: List[str] = ["Question: describe the image. Answer:"],
        targets: List[str] = ["bomb bomb"],
        huggingface_name: str = "Salesforce/blip2-opt-2.7b",
        split: str = "train",
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
        if self.split == "train":
            self.model = Blip2Model.from_pretrained(
                huggingface_name, device_map="auto", torch_dtype=torch.float16
            )
        elif self.split == "eval":
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                huggingface_name, device_map="auto", torch_dtype=torch.float16
            )
        else:
            raise ValueError("Invalid split: {}".format(split))
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)
        assert len(prompts) == len(targets)
        self.batch_size = len(prompts)
        self.prompts = prompts
        self.targets = targets
        self.prompts_then_targets = [
            f"{prompt} {target}" for prompt, target in zip(prompts, targets)
        ]
        self.prompts_then_targets_batch_encoding = self.processor(
            text=self.prompts_then_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # self.targets_batch_encoding = self.processor(
        #     text=targets, return_tensors="pt", padding=True, truncation=True
        # )
        # self.prompts_batch_encoding = self.processor(
        #     text=prompts, return_tensors="pt", padding=True, truncation=True
        # )
        self.prompts_lengths = [
            len(l)
            for l in self.processor(
                text=prompts,
            ).input_ids
        ]
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(torch.clamp(x, min=0.0, max=1.0)).repeat(
            self.batch_size, 1, 1, 1
        )
        inputs = dict(pixel_values=x.to(self.device))
        inputs["input_ids"] = self.prompts_then_targets_batch_encoding.input_ids.to(
            self.device
        )
        inputs["labels"] = self.prompts_then_targets_batch_encoding.input_ids.to(
            self.device
        )
        # Exclude the prompt tokens from the loss computation.
        for batch_idx, prompt_len in enumerate(self.prompts_lengths):
            inputs["labels"][batch_idx, :prompt_len] = -100
        outputs = self.model(**inputs)
        return outputs.loss

    def generate(self, x: torch.Tensor) -> List[str]:
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


class Blip2PredictModel(nn.Module):
    # Maybe should use this to craft adversarial examples instead of previous one
    def __init__(
        self,
        prompt="Question: describe the image. Answer:",
        target_text="bomb bomb",
    ):
        super(Blip2PredictModel, self).__init__()
        self.normalizer = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.device = torch.device("cuda")
        self.eval().requires_grad_(False)
        self.labels = torch.tensor(
            self.processor(text=target_text).input_ids, device=self.device
        ).view(1, -1)
        self.prompt = torch.tensor(
            self.processor(text=prompt).input_ids, device=self.device
        ).view(1, -1)
        self.eval().requires_grad_(False).cuda()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        batch_size = x.shape[0]
        inputs_my = dict(pixel_values=self.normalizer(x).cuda())
        # inputs["input_ids"] = self.prompt.repeat(batch_size, 1)
        inputs = self.processor(images=show_image(x), return_tensors="pt").to(
            self.device
        )
        print(torch.sum((inputs_my["pixel_values"] - inputs["pixel_values"]) ** 2))
        ids = self.model.generate(**inputs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        print(text)
        return text
