import os

import torch.nn
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Tuple


class VLMEnsemble(torch.nn.Module):
    def __init__(
        self,
        model_strs: List[str],
        model_generation_kwargs: Dict[str, Dict[str, Any]],
        accelerator: Accelerator,
    ):
        super().__init__()
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(cuda_visible_devices) >= len(model_strs)
        self.vlms_dict = torch.nn.ModuleDict()
        for model_device_int_str, model_str in enumerate(model_strs):
            # Load BLIP2 models.
            if model_str.startswith("blip2"):
                from old.how_robust_is_bard.src.models.blip2 import (
                    Blip2VisionLanguageModel,
                )

                if model_str.endswith("flan-t5-xxl"):
                    huggingface_name = "Salesforce/blip2-flan-t5-xxl"
                elif model_str.endswith("opt-2.7b"):
                    huggingface_name = "Salesforce/blip2-opt-2.7b"
                elif model_str.endswith("opt-6.7b"):
                    huggingface_name = "Salesforce/blip2-opt-6.7b"
                else:
                    raise ValueError("Invalid model_str: {}".format(model_str))

                vlm = Blip2VisionLanguageModel(
                    huggingface_name=huggingface_name,
                    accelerator=accelerator,
                )

            # Load Instruct BLIP models.
            elif model_str.startswith("instructblip"):
                from old.how_robust_is_bard.src.models.instructblip import (
                    InstructBlipVisionLanguageModel,
                )

                if model_str.endswith("flan-t5-xxl"):
                    huggingface_name = "Salesforce/instructblip-flan-t5-xxl"
                elif model_str.endswith("vicuna-7b"):
                    huggingface_name = "Salesforce/instructblip-vicuna-7b"
                elif model_str.endswith("vicuna-13b"):
                    huggingface_name = "Salesforce/instructblip-vicuna-13b"
                else:
                    raise ValueError("Invalid model_str: {}".format(model_str))

                vlm = InstructBlipVisionLanguageModel(
                    huggingface_name=huggingface_name,
                    accelerator=accelerator,
                )

            elif model_str.startswith("llava"):
                from src.models.llava import LlavaVisionLanguageModel

                vlm = LlavaVisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=model_generation_kwargs[model_str],
                    accelerator=accelerator,
                )

            elif model_str.startswith("prism"):
                from src.models.prismatic import PrismaticVisionLanguageModel

                vlm = PrismaticVisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=model_generation_kwargs[model_str],
                    accelerator=accelerator,
                )

            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            # vlm = self.accelerator.prepare(vlm)
            device_str = f"cuda:{model_device_int_str}"
            self.vlms_dict[model_str] = vlm.to(device_str=device_str)

        self.vlms_dict = torch.nn.ModuleDict(
            {model_str: self.vlms_dict[model_str] for model_str in model_strs}
        )
        # self.vlms_to_attack_dict = self.accelerator.prepare(self.vlms_to_attack_dict)

        self.disable_model_gradients()

    def __len__(self):
        return len(self.vlms_dict)

    def forward(self, args):
        raise NotImplementedError

    def compute_loss(
        self,
        image: torch.Tensor,
        text_data_by_model: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # Always calculate loss per model and use for updating the adversarial example.
        losses_per_model: Dict[str, torch.Tensor] = {}
        total_loss = torch.zeros(1, requires_grad=True, device="cpu")
        for model_idx, (model_name, model_wrapper) in enumerate(self.vlms_dict.items()):
            # Move all tensors to the correct device so we aren't blocked waiting for one model.
            image_on_device = image.to(model_wrapper.device_str, non_blocking=True)
            input_ids_on_device = text_data_by_model[model_name]["input_ids"].to(
                model_wrapper.device_str, non_blocking=True
            )
            attention_mask_on_device = text_data_by_model[model_name][
                "attention_mask"
            ].to(model_wrapper.device_str, non_blocking=True)
            labels_on_device = text_data_by_model[model_name]["labels"].to(
                model_wrapper.device_str, non_blocking=True
            )

            loss_for_model = model_wrapper.compute_loss(
                image=image_on_device,
                input_ids=input_ids_on_device,
                attention_mask=attention_mask_on_device,
                labels=labels_on_device,
            )
            losses_per_model[model_name] = loss_for_model.to("cpu", non_blocking=True)
            if model_name in self.vlms_dict:
                total_loss = total_loss + losses_per_model[model_name]
        avg_loss = total_loss / len(self.vlms_dict)
        losses_per_model["avg"] = avg_loss.to("cpu", non_blocking=True)
        return losses_per_model

    # def compute_loss_parallel(
    #     self, image: torch.Tensor, prompts: List[str], targets: List[str]
    # ) -> Dict[str, torch.Tensor]:
    #     # Always calculate loss per model and use for updating the adversarial example.
    #     losses_per_model: Dict[str, torch.Tensor] = {}
    #     images_replicated = [
    #         image.to(model_wrapper.device_str, non_blocking=True)
    #         for model_wrapper in self.vlms_to_eval_dict.values()
    #     ]
    #     prompts_replicated = [prompts for _ in self.vlms_to_eval_dict]
    #     targets_replicated = [targets for _ in self.vlms_to_eval_dict]
    #     losses_per_model = torch.nn.parallel.parallel_apply(
    #         modules=self.vlms_to_eval_dict.values(),
    #         inputs=(images_replicated, prompts_replicated, targets_replicated),
    #         devices=list(range(len(self.vlms_to_eval_dict))),
    #     )
    #     # avg_loss = total_loss / len(self.vlms_to_attack_dict)
    #     # losses_per_model["avg"] = avg_loss.to("cpu", non_blocking=True)
    #     return losses_per_model

    def disable_model_gradients(self):
        # set all models' requires_grad to False
        for vlm_str, vlm_wrapper in self.vlms_dict.items():
            vlm_wrapper.disable_model_gradients()