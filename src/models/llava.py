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
from src.models.llava_llama_2.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from src.models.llava_llama_2.mm_utils import (
    process_images,
    tokenizer_image_token,
)
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

        if self.huggingface_name == "llava-hf/llava-1.5-7b-hf":
            self.conv_template_name = "vicuna_v1"
        else:
            self.conv_template_name = "default"
        self.conv_template = conversation_templates[self.conv_template_name]

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
        # Based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py#L50
        # and also based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py.
        images = image.repeat(len(prompts), 1, 1, 1).half()

        prompts_with_image_tokens = [
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + prompt
            for prompt in prompts
        ]

        templated_prompts = []
        for prompt in prompts_with_image_tokens:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            templated_prompt = conv.get_prompt()
            templated_prompts.append(templated_prompt)

        input_ids_list: List[List[int]] = [
            tokenizer_image_token(
                templated_prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
            )
            for templated_prompt in templated_prompts
        ]

        # Pad all input_ids to be the same length using the tokenizer's padding token.
        max_length = max([len(input_ids) for input_ids in input_ids_list])
        for idx, input_ids in enumerate(input_ids_list):
            input_ids.extend(
                [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            )
        input_ids_list = torch.tensor(input_ids_list).to(self.device)

        image_pixel_values = self.image_processor(images, return_tensors="pt")[
            "pixel_values"
        ]

        # inputs = self.processor(
        #     images=images,
        #     text=self.prompts_then_targets,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        # ).to(self.device)
        generated_ids = self.model.generate(
            input_ids_list,
            images=image_pixel_values,
            do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
            **self.generation_kwargs,
        )
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return text

    # def wrap_prompts(self, prompts: List[str]):
    #     return LlavaLlama2Prompt(
    #         model=self.model,
    #         tokenizer=self.tokenizer,
    #         text_prompts=prompts,
    #         device=self.device,
    #     )
