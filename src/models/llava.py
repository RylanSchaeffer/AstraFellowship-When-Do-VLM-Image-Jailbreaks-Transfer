import torch
from torchvision import transforms
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
)
from typing import Any, Dict, List

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from src.models.base import VisionLanguageModel
from src.models.conversation import conversation_templates
from src.models.llava_llama_2.prompt_wrapper import (
    prepare_text_prompt,
    LlavaLlama2Prompt,
)


class LlavaVisionLanguageModel(VisionLanguageModel):
    def __init__(
        self,
        huggingface_name: str = "llava-hf/llava-1.5-7b-hf",
        split: str = "train",
        generation_kwargs: Dict[str, Any] = None,
    ):
        super(LlavaVisionLanguageModel, self).__init__()
        self.huggingface_name = huggingface_name
        self.generation_kwargs = generation_kwargs

        self.conversation_template = conversation_templates[self.huggingface_name]

        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(
            model_path=self.huggingface_name,
            model_base=None,
            model_name=get_model_name_from_path(self.huggingface_name),
        )

        self.split = split
        self.device = None

        self.text_prompt_template = prepare_text_prompt("")
        print(self.text_prompt_template)

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

    @torch.no_grad()
    def generate(self, image: torch.Tensor, prompts) -> List[str]:
        # Based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py#L50.
        images = image.repeat(len(prompts), 1, 1, 1).half()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        inputs = self.processor(
            images=images,
            text=self.prompts_then_targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return text

    def wrap_prompts(self, prompts: List[str]) -> LlavaLlama2Prompt:
        return LlavaLlama2Prompt(
            model=self.model,
            tokenizer=self.tokenizer,
            text_prompts=prompts,
            device=self.device,
        )
