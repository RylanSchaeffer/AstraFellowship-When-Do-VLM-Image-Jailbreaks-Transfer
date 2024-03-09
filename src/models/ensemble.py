import torch.nn
from accelerate import Accelerator
from typing import Any, Dict, List, Optional, Tuple


class VLMEnsemble(torch.nn.Module):
    def __init__(
        self,
        model_strs_to_attack: List[str],
        model_strs_to_eval: List[str],
        model_generation_kwargs: Dict[str, Dict[str, Any]],
        accelerator: Accelerator,
    ):
        super().__init__()
        # Confirm that all models to attack will also be evaluated.
        assert all(
            [model_str in model_strs_to_eval for model_str in model_strs_to_attack]
        )
        self.accelerator = accelerator
        self.device = self.accelerator.device

        self.vlms_to_eval_dict = torch.nn.ModuleDict()
        for model_str in model_strs_to_eval:
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
                )

            elif model_str.startswith("llava"):
                from src.models.llava import LlavaVisionLanguageModel

                if model_str.endswith("v1p5-vicuna7b"):
                    huggingface_name = "liuhaotian/llava-v1.5-7b"
                elif model_str.endswith("v1p6-hermes-yi-34b"):
                    huggingface_name = "liuhaotian/llava-v1.6-34b"
                elif model_str.endswith("v1p6-mistral7b"):
                    # Lots of bugs. They needed to be patched here.
                    # https://huggingface.co/Trelis/llava-v1.6-mistral-7b-PATCHED.
                    huggingface_name = "Trelis/llava-v1.6-mistral-7b-PATCHED"
                elif model_str.endswith("v1p6-vicuna7b"):
                    huggingface_name = "liuhaotian/llava-v1.6-vicuna-7b"
                elif model_str.endswith("v1p6-vicuna13b"):
                    huggingface_name = "liuhaotian/llava-v1.6-vicuna-13b"
                else:
                    raise ValueError("Invalid model_str: {}".format(model_str))

                vlm = LlavaVisionLanguageModel(
                    huggingface_name=huggingface_name,
                    generation_kwargs=model_generation_kwargs[model_str],
                )

            elif model_str.startswith("prism"):
                from src.models.prismatic import PrismaticVisionLanguageModel

                vlm = PrismaticVisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=model_generation_kwargs[model_str],
                )

            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            vlm = self.accelerator.prepare(vlm)
            self.vlms_to_eval_dict[model_str] = vlm

        # TODO: Are all of these .prepare() calls necessary?
        self.vlms_to_eval_dict = self.accelerator.prepare(self.vlms_to_eval_dict)

        self.vlms_to_attack_dict = torch.nn.ModuleDict(
            {
                model_str: self.vlms_to_eval_dict[model_str]
                for model_str in model_strs_to_attack
            }
        )
        self.vlms_to_attack_dict = self.accelerator.prepare(self.vlms_to_attack_dict)

        self.disable_model_gradients()

    def __len__(self):
        return len(self.vlms_to_eval_dict)

    def forward(self, args):
        raise NotImplementedError

    def compute_loss(
        self, image: torch.Tensor, prompts: List[str], targets: List[str]
    ) -> Dict[str, torch.Tensor]:
        # Always calculate loss per model and use for updating the adversarial example.
        losses_per_model: Dict[str, torch.Tensor] = {}
        av_target_loss = torch.zeros(1, requires_grad=True)
        for model_idx, (model_name, model_wrapper) in enumerate(
            self.vlm_ensemble.models_to_attack_dict.items()
        ):
            target_loss_for_model = model_wrapper.compute_loss(
                image=image,
                prompts=prompts,
                targets=targets,
            )
            losses_per_model[model_name] = target_loss_for_model.item()
            av_target_loss = av_target_loss + target_loss_for_model.cpu()
        avg_target_loss = av_target_loss / len(self.vlm_ensemble.models_to_attack_dict)
        losses_per_model["avg"] = avg_target_loss
        return losses_per_model

    def disable_model_gradients(self):
        # set all models' requires_grad to False
        for vlm_str, vlm_wrapper in self.vlms_to_eval_dict.items():
            vlm_wrapper.model.requires_grad_(False)
            vlm_wrapper.model.eval()

    def to(self, device: Optional[torch.device] = None):
        if device is None:
            device = self.accelerator.device
        for vlm_str, vlm_wrapper in self.vlms_to_eval_dict.items():
            vlm_wrapper.model = vlm_wrapper.model.to(device)
