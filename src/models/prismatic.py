# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
import lightning
import os.path
from pathlib import Path
from pprint import pprint
import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from prismatic import (
    # available_models,
    # get_model_description,
    load,
)
from prismatic.models.backbones.vision.base_vision import LetterboxPad

from src.models.base import VisionLanguageModel


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100


class PrismaticVisionLanguageModel(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        model_str: str = "prism-dinosiglip+7b",
        generation_kwargs: Dict[str, Any] = None,
        precision: str = "bf16-mixed",
    ):
        super(PrismaticVisionLanguageModel, self).__init__()

        if generation_kwargs is None:
            generation_kwargs = {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_new_tokens": 100,
                "min_new_tokens": 5,
            }

        if model_str == "prism-reproduction-llava-v15+7b":
            model_str = "reproduction-llava-v15+7b"
        elif model_str == "prism-reproduction-llava-v15+13b":
            model_str = "reproduction-llava-v15+13b"

        # self.accelerator = accelerator
        self.model_str = model_str
        self.generation_kwargs = generation_kwargs

        hf_token = (
            Path(os.path.expandvars("$LFS_HOME/.cache/huggingface/token"))
            .read_text()
            .strip()
        )

        # # This incorrectly uses the data structure returned by available_model_names.
        # if model_str not in available_models():
        #     pprint(available_models())
        #     raise ValueError(f"Invalid model_str: {model_str}")

        self.model = load(model_id_or_path=model_str, hf_token=hf_token)
        self.precision_str = precision
        if self.precision_str in {"bf16-mixed", "bf16-true"}:
            self.precision_dtype = torch.bfloat16
        elif self.precision_str == "16-true":
            self.precision_dtype = torch.float16
        elif self.precision_str in {"32", "32-true"}:
            self.precision_dtype = torch.float32
        elif self.precision_str in {"64", "64-true"}:
            self.precision_dtype = torch.float64
        else:
            raise ValueError(f"Invalid precision: {self.precision_str}")

        # The llm_backbone.llm has dtype float32. Switch to bfloat16.
        self.model.llm_backbone.llm = self.model.llm_backbone.llm.to(
            self.precision_dtype
        )
        self.images_transform_fn = self.create_images_transform_fn(model_str)

    def create_images_transform_fn(self, model_str: str) -> Callable:
        if "dinosiglip" in model_str:
            # Convert to float32, then remove the ToTensor transform because that is applicable to PIL Images.
            # TODO: Decide what to do about that LetterboxPadding.
            dino_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.ConvertImageDtype(self.precision_dtype)]
                + [
                    t
                    for t in self.model.vision_backbone.image_transform.dino_image_transform.transforms
                    if not isinstance(t, torchvision.transforms.ToTensor)
                    and not isinstance(t, LetterboxPad)
                ]
            )

            # TODO: Decide what to do about that LetterboxPadding.
            siglip_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.ConvertImageDtype(self.precision_dtype)]
                + [
                    t
                    for t in self.model.vision_backbone.image_transform.siglip_image_transform.transforms
                    if not isinstance(t, torchvision.transforms.ToTensor)
                    and not isinstance(t, LetterboxPad)
                ]
            )

            def images_transform_fn(images: torch.Tensor) -> Dict[str, torch.Tensor]:
                dino_features = dino_transforms(images)
                siglip_features = siglip_transforms(images)
                # TODO: Figure out why (what I think is) the same code is sometimes returning a list and other times returning a tensor.
                if isinstance(dino_features, list):
                    assert len(dino_features) == 1
                    dino_features = dino_features[0]
                if isinstance(siglip_features, list):
                    assert len(siglip_features) == 1
                    siglip_features = siglip_features[0]
                transformed_images = {
                    "dino": dino_features,
                    "siglip": siglip_features,
                }
                return transformed_images

        else:
            # Convert to float32, then remove the ToTensor transform because that is applicable to PIL Images.
            # TODO: Decide what to do about that LetterboxPadding.
            transforms = torchvision.transforms.Compose(
                [torchvision.transforms.ConvertImageDtype(self.precision_dtype)]
                + [
                    t
                    for t in self.model.vision_backbone.image_transform.transforms
                    if not isinstance(t, torchvision.transforms.ToTensor)
                ]
            )

            def images_transform_fn(images: torch.Tensor) -> torch.Tensor:
                transformed_images = transforms(images)
                return transformed_images

        return images_transform_fn

    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = image.repeat(len(input_ids), 1, 1, 1)
        transformed_images: Union[
            torch.Tensor, Dict[str, torch.Tensor]
        ] = self.images_transform_fn(images)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=transformed_images,
        )
        return outputs.loss

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if targets is None:
            targets = [None for _ in range(len(prompts))]

        prompt_texts = []
        for prompt, target in zip(prompts, targets):
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt)
            if target is not None:
                prompt_builder.add_turn(role="gpt", message=target)
            prompt_text = prompt_builder.get_prompt()
            prompt_texts.append(prompt_text)

        batch_encoding = self.model.llm_backbone.tokenizer(
            prompt_texts, padding=True, return_tensors="pt"
        )
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]

        results = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        pad_token = self.model.llm_backbone.tokenizer.special_tokens_map["pad_token"]
        pad_token_input_id = self.model.llm_backbone.tokenizer.convert_tokens_to_ids(
            pad_token
        )

        if targets[0] is not None:
            labels = input_ids.clone()
            last_nonpadding_indices = torch.argmin(
                (labels != pad_token_input_id).float(), axis=1
            )
            # If there are no padding tokens, then we want to set the last non-padding index to the length.
            last_nonpadding_indices[last_nonpadding_indices == 0] = labels.shape[1]

            # Find the last non-zero token. Then set labels to ignore for anything
            # before and before the targets (plus two).
            tokenized_labels = self.model.llm_backbone.tokenizer(targets).input_ids
            for batch_idx, (last_nonpadding_idx, tokenized_label) in enumerate(
                zip(last_nonpadding_indices, tokenized_labels)
            ):
                target_start_idx = last_nonpadding_idx - len(tokenized_label) - 1
                labels[batch_idx, :target_start_idx] = IGNORE_INDEX
                # Exclude the EOS token, if it exists.
                if (
                    labels[batch_idx, last_nonpadding_idx - 1]
                    == self.model.llm_backbone.tokenizer.eos_token_id
                ):
                    labels[batch_idx, last_nonpadding_idx - 1] = IGNORE_INDEX

            # Also mask out the padding tokens.
            labels[labels == pad_token_input_id] = IGNORE_INDEX
            results["labels"] = labels

        return results

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        assert image.shape[0] == 1
        pil_image = torchvision.transforms.functional.to_pil_image(
            image[0].to(torch.float32)
        )
        model_generations = []
        # Currently, Prismatic only supports one prompt at a time.
        # See https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L535
        for prompt in prompts:
            # Build prompt
            prompt_builder = self.model.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=prompt)
            prompt_text = prompt_builder.get_prompt()
            # prompt_text = self.accelerator.prepare(prompt_text)
            generated_text = self.model.generate(
                prompt_text=prompt_text,
                image=pil_image,  # This needs to be the PIL image.
                do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
                **self.generation_kwargs,
            )
            model_generations.append(generated_text)
        return model_generations

    def disable_model_gradients(self):
        self.model.llm_backbone.requires_grad_(False)
        self.model.llm_backbone.eval()
        self.model.projector.requires_grad_(False)
        self.model.projector.eval()
        self.model.vision_backbone.requires_grad_(False)
        self.model.vision_backbone.eval()
        self.model.eval()

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        non_blocking: bool = False,
    ):
        if device is not None:
            self.model.llm_backbone.llm = self.model.llm_backbone.llm.to(device=device)
            self.model.llm_backbone = self.model.llm_backbone.to(device=device)
            self.model.projector = self.model.projector.to(device=device)
            self.model.vision_backbone = self.model.vision_backbone.to(device=device)
            self.model = self.model.to(device=device)
            super(PrismaticVisionLanguageModel, self).to(device=device)
        if dtype is not None:
            self.model.llm_backbone.llm = self.model.llm_backbone.llm.to(dtype=dtype)
            self.model.llm_backbone = self.model.llm_backbone.to(dtype=dtype)
            self.model.projector = self.model.projector.to(dtype=dtype)
            self.model.vision_backbone = self.model.vision_backbone.to(dtype=dtype)
            self.model = self.model.to(dtype=dtype)
            self.precision_dtype = dtype
            super(PrismaticVisionLanguageModel, self).to(dtype=dtype)

        return self
