# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.xgen_utils.image_processing_blip_3 import Blip3ImageProcessor
from src.models.label_compute import make_labels
from transformers import AutoTokenizer, StoppingCriteria
import lightning
import torch
from typing import Any, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizer

from src.models.base import VisionLanguageModel

from src.models.xgen_utils.modeling_blip_3 import Blip3ModelForConditionalGeneration
from src.models.xgen_utils.utils import (
    apply_xgen_prompt_template,
    apply_xgen_prompt_template_with_target,
)


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100

def make_labels_xgen(
    input_ids: torch.Tensor, pad_token_id: int, targets: list[str], tokenizer
):
    labels = input_ids.clone()
    last_nonpadding_indices = torch.argmin((labels != pad_token_id).float(), axis=1)
    # print(f"{last_nonpadding_indices=}")
    # If there are no padding tokens, then we want to set the last non-padding index to the length.
    last_nonpadding_indices[last_nonpadding_indices == 0] = (
        labels.shape[1] - 1
    )  # Minus one!!
    # print(f"{last_nonpadding_indices=}")

    # Find the last non-zero token. Then set labels to ignore for anything
    # before and before the targets (plus two).
    tokenized_labels = tokenizer(targets).input_ids
    for batch_idx, (last_nonpadding_idx, tokenized_label) in enumerate(
        zip(last_nonpadding_indices, tokenized_labels)
    ):
        # + 2 that it does not incldue the assistant tag.
        # TODO: check prism models?
        target_start_idx = last_nonpadding_idx - len(tokenized_label) + 2
        # print(f"{target_start_idx=}")
        labels[batch_idx, :target_start_idx] = IGNORE_INDEX

    # Also mask out the padding tokens.
    labels[labels == pad_token_id] = IGNORE_INDEX
    return labels



def pad_and_make_attention_masks(
    input_ids: list[list[int]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """
    Left pads the input_ids with pad_token_id and creates the attention masks.
    """
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = [
        [pad_token_id] * (max_len - len(ids)) + ids for ids in input_ids
    ]
    attention_masks = [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in input_ids]
    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(attention_masks),
    }


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids


class XgenVisionLanguageModel(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        model_str: str = "xgen-mm-phi3-mini-instruct-r-v1",
        generation_kwargs: Mapping[str, Any] | None = None,
        precision: str = "bf16-mixed",
    ):
        super().__init__()

        if generation_kwargs is None:
            generation_kwargs = {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_new_tokens": 100,
                "min_new_tokens": 5,
            }

        self.model_str = model_str
        self.generation_kwargs = generation_kwargs

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

        model_path = f"Salesforce/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using Salesforce model: {model_path}")

        # qwen doesn't have a specific pad token, but since we mask it out we can use any token
        # see https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md

        self.model: Blip3ModelForConditionalGeneration = (
            Blip3ModelForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.precision_dtype,
            ).to(self.precision_dtype)
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
        self.model.update_special_tokens(self.tokenizer)
        self.image_processor = Blip3ImageProcessor.from_pretrained(model_path)

        self.already_logged_new_mask: bool = False  # For print debugigng
        self.already_logged_text: bool = False  # For print debugigng
        self.pad_token_id = self.tokenizer.pad_token_id
        assert self.pad_token_id is not None, "Expected pad token id to be set."

    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,  # before adding image tokens, because this model needs the image_seq_mask
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = self.model.device

        # Since we only get a single image, we need to repeat it for the batch size.
        assert image.ndim == 4, f"Expected 4 dims, got {image.ndim}"
        # assert that we only have one image here
        assert (
            image.size(0) == 1
        ), f"Expected only 1 image that we repeat, got {image.size(0)}"

        # image_embeds = image_embeds.repeat(bs, 1, 1)
        image_inputs = self.image_processor(
            image, return_tensors="pt", image_aspect_ratio="anyres"
        )["pixel_values"]
        # we'll get back [1, 1, 5, 3, 378, 378], we need to repeat to get [bs, 1, 5, 3, 378, 378]
        bs = input_ids.size(0)
        repeated =image_inputs.repeat(bs, 1, 1, 1, 1, 1).to(self.precision_dtype)

        merged_inputs = {
            "pixel_values": repeated,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        final_inputs = {k: v.to(device) for k, v in merged_inputs.items()}

        outputs = self.model.vlm(
            **final_inputs,
        )
        return outputs.loss

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert targets is not None, "Not support yet."

        prompt_texts = [
            # make context adds the assistant token in to continue
            apply_xgen_prompt_template_with_target(prompt=prompt, target=target)
            for prompt, target in zip(prompts, targets)
        ]
        pad_token_id = self.pad_token_id
        assert pad_token_id is not None, "Expected pad token id to be set."

        results = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        input_ids = results["input_ids"]
        attention_mask = results["attention_mask"]
        if targets[0] is not None:
            labels = make_labels_xgen(
                input_ids=input_ids,  # type: ignore
                pad_token_id=pad_token_id,
                targets=targets,
                tokenizer=self.tokenizer,
            )
            results["labels"] = labels

        if not self.already_logged_text:
            torch.set_printoptions(threshold=10000)
            # first_text = prompt_texts[0]
            # print(f"First text: {first_text}")
            print(f"First input_ids: {input_ids[0]}")
            print(f"First attention_mask: {attention_mask[0]}")
            print(f"First labels: {results['labels'][0]}")
            if len(input_ids) > 1:
                print(f"Second input ids: {input_ids[1]}")
                print(f"Second attention_mask: {attention_mask[1]}")
                print(f"Second labels: {results['labels'][1]}")
            # non_minus_100 = [r for r in results["labels"][0] if r != IGNORE_INDEX]
            # non_minus_100_text = self.tokenizer.decode(non_minus_100)
            # print(f"Example text that we calculate loss on: {non_minus_100_text}")
            torch.set_printoptions(profile="default")
            self.already_logged_text = True

        return results

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        assert image.shape[0] == 1
        assert image.ndim == 4, f"Expected (1, 3, H, W), got {image.shape}"

        # get (H, W) from (1, 3, H, W)
        image_size = [tuple(image.shape[2:])]
        # we have (1, 3, h,w) , we want (3, H, W)
        model_generations = []
        device = self.model.device
        image_inputs = self.image_processor(
            image, return_tensors="pt", image_aspect_ratio="anyres"
        ).to(self.precision_dtype)

        for prompt in prompts:
            new_prompt = apply_xgen_prompt_template(prompt)
            # print(f"Prompting the model with: {new_prompt}")
            language_inputs = self.tokenizer([new_prompt], return_tensors="pt")
            merged_inputs = {**image_inputs, **language_inputs}
            final_inputs = {k: v.to(device) for k, v in merged_inputs.items()}

            do_sample = (
                True if self.generation_kwargs.get("temperature", 0) > 0 else False
            )

            # # run the model to get the response

            # print(f"Prompting with image: {image}")

            generation_config = self.model.generation_config
            assert (
                generation_config is not None
            ), "Expected generation config to be set."

            generated_text = self.model.generate(
                **final_inputs,
                image_size=image_size,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                stopping_criteria=[EosListStoppingCriteria()],
                **self.generation_kwargs,
            )
            # print(f"Got type: {type(outputs)}")
            out = self.tokenizer.decode(
                generated_text.squeeze(), skip_special_tokens=True
            ).split("<|end|>")[0]

            model_generations.append(out)

        return model_generations

    def disable_model_gradients(self):
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.vlm.requires_grad_(False)
        self.model.vlm.eval()
        self.model.vlm.vision_encoder.requires_grad_(False)
        self.model.vlm.vision_encoder.eval()
        self.model.vlm.lang_model.requires_grad_(False)
        self.model.vlm.lang_model.eval()
        
        

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        non_blocking: bool = False,
    ):
        if device is not None:
            self.model = self.model.to(device=device)
            self.model.vlm = self.model.vlm.to(device=device)
            self.model.vlm.vision_encoder = self.model.vlm.vision_encoder.to(device=device)
            self.model.vlm.lang_model = self.model.vlm.lang_model.to(device=device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
            self.precision_dtype = dtype

        return self
