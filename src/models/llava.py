import torch
from torchvision import transforms
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
)
from typing import Any, Dict, List, Optional, Tuple

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
from src.models.llava_llama_2.visual_attacker import normalize


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
        # Based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py#L50
        # and also based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py.
        images = image.repeat(len(prompts), 1, 1, 1)
        # image_pixel_values = self.image_processor(
        #     images, do_rescale=False, return_tensors="pt"
        # )["pixel_values"].half()
        image_pixel_values = normalize(images).half()

        results = (
            self.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
                prompts=prompts,
                targets=targets,
            )
        )
        for k, v in results.items():
            results[k] = v.to(self.device)

        outputs = self.model(
            input_ids=results["input_ids"],
            attention_mask=results["attention_mask"],
            labels=results["labels"],
            images=image_pixel_values,
        )
        return outputs.loss

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if targets is None:
            targets = [None for _ in range(len(prompts))]

        prompts_with_image_tokens = [
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + prompt
            for prompt in prompts
        ]

        templated_prompts = []
        for prompt, target in zip(prompts_with_image_tokens, targets):
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], target)
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
        attention_mask = []
        max_length = max([len(input_ids) for input_ids in input_ids_list])
        for idx, input_ids in enumerate(input_ids_list):
            padding_length = max_length - len(input_ids)
            attention_mask.append(
                [1 for _ in range(max_length - padding_length)]
                + [0 for _ in range(padding_length)]
            )
            input_ids.extend(
                [self.tokenizer.pad_token_id for _ in range(padding_length)]
            )
        input_ids = torch.tensor(input_ids_list)
        attention_mask = torch.tensor(attention_mask)

        results = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if targets[0] is not None:
            labels = input_ids.clone()
            last_nonpadding_indices = torch.argmin((labels != 0).float(), axis=1)

            # Find the last non-zero token. Then set labels to ignore for anything
            # before and before the targets (plus two).
            tokenized_labels = self.tokenizer(targets).input_ids
            for batch_idx, (last_nonpadding_idx, tokenized_label) in enumerate(
                zip(last_nonpadding_indices, tokenized_labels)
            ):
                target_start_idx = last_nonpadding_idx - len(tokenized_label) - 1
                labels[batch_idx, :target_start_idx] = IGNORE_INDEX

            # Also mask out the padding tokens.
            labels[labels == 0] = IGNORE_INDEX
            results["labels"] = labels

        return results

    @torch.no_grad()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # Based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/run_llava.py#L50
        # and also based on https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa.py.
        images = image.repeat(len(prompts), 1, 1, 1).half()
        image_pixel_values = self.image_processor(
            images, do_rescale=False, return_tensors="pt"
        )["pixel_values"]

        input_ids = (
            self.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
                prompts=prompts,
                targets=None,
            )["input_ids"].to(self.device)
        )

        generated_ids = self.model.generate(
            input_ids,
            images=image_pixel_values.half(),
            do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
            **self.generation_kwargs,
        )
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return text
