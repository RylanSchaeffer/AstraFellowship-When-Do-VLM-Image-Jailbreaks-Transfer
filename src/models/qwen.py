# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
from deepseek_vl.models.image_processing_vlm import (
    VLMImageProcessor,
    VLMImageProcessorConfig,
)
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel
from transformers import AutoImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning
import torch
from typing import Any, Callable, Dict, List, Mapping, Optional
from transformers import PreTrainedTokenizer

from src.models.base import VisionLanguageModel
from deepseek_vl.models import (
    VLChatProcessor,
    MultiModalityCausalLM,
)
from deepseek_vl.models.processing_vlm import (
    VLChatProcessorOutput,
)
from src.models.qwen_utils.qwen_generation_utils import make_context_assistant_completion, make_context_assistant_target

from src.models.qwen_utils.visual import VisionTransformer


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100


def add_image_token(
    input_ids: torch.Tensor,  # (seq_len) non batched
    pixel_values: torch.Tensor,  # single image
    processor: VLChatProcessor,
) -> VLChatProcessorOutput:
    image_token_mask: torch.BoolTensor = input_ids == processor.image_id
    image_indices = image_token_mask.nonzero()
    input_ids, num_image_tokens = processor.add_image_token(
        image_indices=image_indices,
        input_ids=input_ids,
    )

    assert pixel_values.ndim == 4
    # remove the batch dim
    # we have (1, 3, h,w) , we want (3, H, W)
    new_image = pixel_values.squeeze(0)

    image_output = processor.image_processor.preprocess_one(
        image=new_image, return_tensors="pt"
    )
    prepare = VLChatProcessorOutput(
        sft_format="placeholder",
        input_ids=input_ids,
        pixel_values=image_output,
        num_image_tokens=num_image_tokens,
    )
    return prepare


def to_deepseek_sft_format(
    prompt: str,
    target: str,
    processor: VLChatProcessor,
) -> str:
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{prompt}",
            "images": ["doesn't matter"],
        },
        {"role": "Assistant", "content": f"{target}"},
    ]
    return processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt=processor.system_prompt,
    )


class QwenVisionLanguageModel(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        model_str: str = "Qwen-VL-Chat",
        generation_kwargs: Mapping[str, Any] | None= None,
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
        
        model_path = f"Qwen/{model_str}"

        # not sure why we need to register the image processor manually
        print(f"Using Qwen model: {model_path}")

        # self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)  # type: ignore
        self.model: QWenLMHeadModel = QWenLMHeadModel.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
        ).to(self.precision_dtype)
        self.vision_model: VisionTransformer = self.model.transformer.visual
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        # eos is the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
        transformed_image: torch.Tensor = self.vision_model.image_transform(image)
        image_embeds = self.vision_model(transformed_image)


        outputs = self.model(
            input_ids=input_ids,
            image_embeds=image_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
        self,
        prompts: List[str],
        targets: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert targets is not None, "Not support yet."

        ### FIX ME:: NEEDS THE HSITORY UHHH
        prompt_texts = [
            # make context adds the assistant token in to continue
            make_context_assistant_target(tokenizer=self.tokenizer, query=prompt, target=target)
            for prompt, target in zip(prompts, targets)
        ]

        batch_encoding = self.tokenizer(prompt_texts, padding=True, return_tensors="pt")
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]

        results = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        pad_token_input_id: int = self.tokenizer.eos_token_id  # type: ignore

        if targets[0] is not None:
            labels = make_labels(
                input_ids=input_ids,
                pad_token_id=pad_token_input_id,
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
            print(f"First labels: {results["labels"][0]}")
            if len(input_ids) > 1:
                print(f"Second input ids: {input_ids[1]}")
                print(f"Second attention_mask: {attention_mask[1]}")
                print(f"Second labels: {results["labels"][1]}")
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
            context = make_context_assistant_completion(tokenizer=self.tokenizer, query=prompt)
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{prompt}",
                    "images": ["doesn't matter"],
                },
                {"role": "Assistant", "content": ""},
            ]

            # todo: tihs can be batched
            input_1 = self.processor(
                conversations=conversation, images=image, force_batchify=True
            ).to(self.model.device)

            # run image encoder to get the image embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**input_1)
            do_sample = (
                True if self.generation_kwargs.get("temperature", 0) > 0 else False
            )
            # # run the model to get the response
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                # attention_mask=inputs_embeds.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # max_new_tokens=512,
                do_sample=do_sample,
                **self.generation_kwargs,
                use_cache=True,
            )
            out: str = self.tokenizer.decode(
                outputs.squeeze(), skip_special_tokens=True
            )

            model_generations.append(out)

        return model_generations

    def disable_model_gradients(self):
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.vision_model.requires_grad_(False)
        self.model.vision_model.eval()
        self.model.aligner.requires_grad_(False)
        self.model.aligner.eval()
        self.model.language_model.requires_grad_(False)
        self.model.language_model.eval()

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
