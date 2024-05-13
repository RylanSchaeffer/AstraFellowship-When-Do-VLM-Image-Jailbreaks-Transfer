# Based on the models at https://github.com/TRI-ML/prismatic-vlms?tab=readme-ov-file.
import lightning
import torch
import torchvision
import torchvision.transforms
import torchvision.transforms.functional
from typing import Any, Callable, Dict, List, Optional

from src.models.base import VisionLanguageModel
from deepseek_vl.models import (
    VLChatProcessor,
    MultiModalityCausalLM,
)
from deepseek_vl.models.processing_vlm import (
    VLChatProcessorOutput,
)


# Labels with these indices will be ignored by cross entropy loss in PyTorch.
IGNORE_INDEX = -100

# def process_batch(
#         conversations: list[list[dict[str, str]]],
#         processor: VLChatProcessor,
#         images: list[Image] = None,
#     ):
#         """
#         Custom function because the deepseek boys don't have a nice way to process batches wtf

#         Args:
#             conversations (List[Dict]): conversations with a list of messages;
#             images (List[ImageType]): the list of images;
#             **kwargs:

#         Returns:
#             outputs (BaseProcessorOutput): the output of the processor,
#                 - input_ids (torch.LongTensor): [N + image tokens]
#                 - target_ids (torch.LongTensor): [N + image tokens]
#                 - images (torch.FloatTensor): [n_images, 3, H, W]
#                 - image_id (int): the id of the image token
#                 - num_image_tokens (List[int]): the number of image tokens
#         """

#         # apply sft format
#         processor.tokenizer.pad_token=processor.tokenizer.eos_token
#         formatted = [processor.apply_sft_template_for_multi_turn_prompts(
#             conversations=c,
#             sft_format=processor.sft_format,
#             system_prompt=processor.system_prompt,
#         ) for c in conversations]


#         # tokenize
#         tokenizer_result = processor.tokenizer(formatted, padding=True, return_tensors="pt")


#         # add image tokens to the input_ids
#         image_token_mask: torch.BoolTensor = tokenizer_result == processor.image_id
#         image_indices = image_token_mask.nonzero()

#         input_ids, num_image_tokens = processor.add_image_token(
#             image_indices=image_indices,
#             input_ids=tokenizer_result,
#         )

#         # load images
#         images_outputs = processor.image_processor(images, return_tensors="pt")

#         prepare = VLChatProcessorOutput(
#             sft_format="placeholder",
#             input_ids=input_ids,
#             pixel_values=images_outputs.pixel_values,
#             num_image_tokens=num_image_tokens,
#         )

#         return prepare


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
    
    # we need to preprocess the image to pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB
    pil_img = torchvision.transforms.functional.to_pil_image(
        pixel_values[0].to(torch.float32)
    )

    image_output = processor.image_processor([pil_img], return_tensors="pt")
    prepare = VLChatProcessorOutput(
        sft_format="placeholder",
        input_ids=input_ids,
        pixel_values=image_output.pixel_values,
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


class DeepSeekVisionLanguageModel(VisionLanguageModel, lightning.LightningModule):
    def __init__(
        self,
        model_str: str = "deepseek-vl-1.3b-chat",
        generation_kwargs: Dict[str, Any] = None,
        precision: str = "bf16-mixed",
    ):
        # https://huggingface.co/HuggingFaceM4/idefics2-8b
        super(DeepSeekVisionLanguageModel, self).__init__()

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
        model_path = f"deepseek-ai/{model_str}"
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.model = MultiModalityCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.precision_dtype,
        ).to(self.precision_dtype)
        self.tokenizer = self.processor.tokenizer
        # eos is the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        # images = image.repeat(len(input_ids), 1, 1, 1)

        # outputs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels,
        #     pixel_values=transformed_images,
        # )
        processor_outputs: list[VLChatProcessorOutput] = [
            add_image_token(
                input_ids=input_id, pixel_values=image, processor=self.processor
            )
            for input_id in input_ids
        ]
        batched = self.processor.batchify(processor_outputs).to(self.device)

        # call batchify because it adds the image masks that we need

        inputs_embeds = self.model.prepare_inputs_embeds(**batched).to(self.device)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
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

        prompt_texts = [
            to_deepseek_sft_format(prompt, target, self.processor)
            for prompt, target in zip(prompts, targets)
        ]
        first_text = prompt_texts[0]
        print(f"First text: {first_text}")

        batch_encoding = self.tokenizer(
            prompt_texts, padding=True, return_tensors="pt"
        )
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]

        results = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        pad_token_input_id = self.tokenizer.eos_token_id

        if targets[0] is not None:
            labels = input_ids.clone()
            last_nonpadding_indices = torch.argmin(
                (labels != pad_token_input_id).float(), axis=1
            )
            # If there are no padding tokens, then we want to set the last non-padding index to the length.
            last_nonpadding_indices[last_nonpadding_indices == 0] = labels.shape[1]

            # Find the last non-zero token. Then set labels to ignore for anything
            # before and before the targets (plus two).
            tokenized_labels = self.tokenizer(targets).input_ids
            for batch_idx, (last_nonpadding_idx, tokenized_label) in enumerate(
                zip(last_nonpadding_indices, tokenized_labels)
            ):
                # + 1 that it does not incldue the assistant tag.
                # TODO: check prism models?
                target_start_idx = last_nonpadding_idx - len(tokenized_label) + 1
                labels[batch_idx, :target_start_idx] = IGNORE_INDEX

            # Also mask out the padding tokens.
            labels[labels == pad_token_input_id] = IGNORE_INDEX
            results["labels"] = labels

        print(f"First input_ids: {input_ids[0]}")
        print(f"First attention_mask: {attention_mask[0]}")
        print(f"First labels: {results["labels"][0]}")
        non_minus_100 = [r for r in results["labels"][0] if r != IGNORE_INDEX]
        non_minus_100_text = self.tokenizer.decode(non_minus_100)
        print(f"Example text that we calculate loss on: {non_minus_100_text}")

        return results

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, prompts: List[str]) -> List[str]:
        # We should only have a single image.
        assert image.shape[0] == 1
        pil_image = torchvision.transforms.functional.to_pil_image(
            image[0].to(torch.float32)
        )
        model_generations = []
        for prompt in prompts:
            # Create inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_template_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=chat_template_prompt, images=[pil_image], return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True if self.generation_kwargs["temperature"] > 0 else False,
                **self.generation_kwargs,
            )
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            # Strip off the user's request.
            generated_text = generated_text[len(chat_template_prompt) :]
            model_generations.append(generated_text)

        return model_generations

    def disable_model_gradients(self):
        self.model.requires_grad_(False)
        self.model.eval()

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        non_blocking: bool = False,
    ):
        if device is not None:
            self.model = self.model.to(device=device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)
            self.precision_dtype = dtype

        return self
