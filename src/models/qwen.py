# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from src.models.label_compute import make_labels
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoTokenizer
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizer

from src.models.base import VisionLanguageModel
from src.models.qwen_utils.qwen_generation_utils import (
    get_stop_words_ids,
    make_context_assistant_completion,
    make_context_assistant_target,
)
from src.models.qwen_utils.qwen_load import only_assistant_response

from src.models.qwen_utils.visual import VisionTransformer


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100


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


class QwenVisionLanguageModel(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        model_str: str = "Qwen-VL-Chat",
        generation_kwargs: Mapping[str, Any] | None = None,
        precision: str = "bf16-mixed",
        device: torch.device | str | None = None,
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

        model_path = f"Qwen/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using Qwen model: {model_path}")

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)  # type: ignore
        # qwen doesn't have a specific pad token, but since we mask it out we can use any token
        # see https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md

        self.pad_token_id = 55
        device_map = device if device is not None else None
        self.model: QWenLMHeadModel = QWenLMHeadModel.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
            device_map=device_map,
        ).to(self.precision_dtype)
        self.vision_model: VisionTransformer = self.model.transformer.visual

        self.already_logged_new_mask: bool = False  # For print debugigng
        self.already_logged_text: bool = False  # For print debugigng

    def create_images_transform_fn(self, model_str: str) -> Callable:
        raise NotImplementedError(
            "create_images_transform_fn is not implemented for DeepSeek models."
        )

    def compute_loss(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,  # before adding image tokens, because this model needs the image_seq_mask
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        device = self.model.device

        # Since we only get a single image, we need to repeat it for the batch size.
        bs = input_ids.size(0)
        assert image.ndim == 4, f"Expected 4 dims, got {image.ndim}"
        # assert that we only have one image here
        assert (
            image.size(0) == 1
        ), f"Expected only 1 image that we repeat, got {image.size(0)}"

        image_embeds: torch.Tensor = self.vision_model.transform_and_forward(
            image.to(device=device)
        )
        # bs, num_image_tokens, dim
        assert image_embeds.ndim == 3, f"Expected 3 dims, got {image_embeds.ndim}"

        image_embeds = image_embeds.repeat(bs, 1, 1)

        outputs = self.model(
            input_ids=input_ids.to(device=device),
            image_embeds=image_embeds.to(device=device),
            attention_mask=attention_mask.to(device=device),
            labels=labels.to(device=device),
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
            make_context_assistant_target(
                tokenizer=self.tokenizer,
                query=self.tokenizer.from_list_format(
                    [  # type: ignore
                        {"image": "image_url"},  # needed to make them image tokens
                        {"text": prompt},
                    ]
                ),
                target=target,
            )
            for prompt, target in zip(prompts, targets)
        ]
        pad_token_id = self.pad_token_id
        assert pad_token_id is not None, "Expected pad token id to be set."
        results = pad_and_make_attention_masks(
            input_ids=[self.tokenizer.encode(text) for text in prompt_texts],
            pad_token_id=pad_token_id,
        )
        input_ids = results["input_ids"]
        attention_mask = results["attention_mask"]
        if targets[0] is not None:
            labels = make_labels(
                input_ids=input_ids,
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
        # we have (1, 3, h,w) , we want (3, H, W)
        model_generations = []
        for prompt in prompts:
            new_prompt = self.tokenizer.from_list_format(
                [  # type: ignore
                    {"image": "image_url"},  # needed to make them image tokens
                    {"text": prompt},
                ]
            )
            # print(f"Prompting the model with: {new_prompt}")
            context: list[int] = make_context_assistant_completion(
                tokenizer=self.tokenizer, query=new_prompt
            )
            input_ids = torch.tensor(context).unsqueeze(0).to(self.model.device)

            do_sample = (
                True if self.generation_kwargs.get("temperature", 0) > 0 else False
            )

            # # run the model to get the response

            # print(f"Prompting with image: {image}")

            generation_config = self.model.generation_config
            assert (
                generation_config is not None
            ), "Expected generation config to be set."
            # # run the model to get the response
            # these stop words are the im_end, so they are the REAL eos
            stop_words = get_stop_words_ids(generation_config.chat_format, self.tokenizer)  # type: ignore
            outputs = self.model.generate(
                inputs=input_ids,
                images=image,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # max_new_tokens=512,
                do_sample=do_sample,
                use_cache=True,
                stop_words_ids=stop_words,
                **self.generation_kwargs,
            )
            # print(f"Got type: {type(outputs)}")
            out: str = self.tokenizer.decode(
                outputs.squeeze(), skip_special_tokens=True
            )
            clean_out = only_assistant_response(initial_prompt=prompt, response=out)

            model_generations.append(clean_out)

        return model_generations

    def disable_model_gradients(self):
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.transformer.requires_grad_(False)
        self.model.transformer.eval()
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        non_blocking: bool = False,
    ):
        if device is not None:
            self.model: QWenLMHeadModel = self.model.to(device=device)
            self.model.lm_head = self.model.lm_head.to(device=device)
            self.model.transformer = self.model.transformer.to(device=device)
            # No idea why we need to do this, shouldn't the MultiModalityCausalLM.to already do this???
            # print(f"moving the vision model to {device}")
            # self.model.vision_model = self.model.vision_model.to(device=device)
            # self.model.aligner = self.model.aligner.to(device=device)
            # self.model.language_model = self.model.language_model.to(device=device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
            self.precision_dtype = dtype

        return self
