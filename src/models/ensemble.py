import numpy as np
import os
import lightning
import torch.nn
from typing import Any, Dict, List, Optional, Tuple


from src.models.base import VisionLanguageModel


class VLMEnsemble(lightning.LightningModule):
    def __init__(
        self,
        model_strs: List[str],
        model_generation_kwargs: Dict[str, Dict[str, Any]],
        precision: str = "bf16-mixed",
    ):
        super().__init__()
        self.vlms_dict: Dict[str, VisionLanguageModel] = {}

        device_count = torch.cuda.device_count()
        assert device_count > 0, "No CUDA devices available."
        assert device_count >= len(model_strs), "Need at least one GPU per VLM."

        for model_device_int_str, model_str in enumerate(model_strs):
            # Enable overwriting default generation kwargs.
            if model_str in model_generation_kwargs:
                generation_kwargs = model_generation_kwargs[model_str]
            else:
                generation_kwargs = None

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

            elif model_str.startswith("idefics2"):
                from src.models.idefics2 import Idefics2VisionLanguageModel

                vlm = Idefics2VisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=generation_kwargs,
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

                vlm = LlavaVisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=generation_kwargs,
                )

            elif model_str.startswith("prism"):
                from src.models.prismatic import PrismaticVisionLanguageModel

                vlm = PrismaticVisionLanguageModel(
                    model_str=model_str,
                    generation_kwargs=generation_kwargs,
                    precision=precision,
                )

            else:
                raise ValueError("Invalid model_str: {}".format(model_str))

            if torch.cuda.is_available():
                # If we have N GPUs, we want the first to go to GPU N-1, the second to GPU N-2, etc.
                # TODO: Was this actually because someone else was using the GPUs at the same time?
                vlm = vlm.to(
                    torch.device(f"cuda:{device_count - model_device_int_str - 1}")
                )

            self.vlms_dict[model_str] = vlm

        self.disable_vlm_gradients()

    def __len__(self):
        return len(self.vlms_dict)

    def compute_loss(
        self,
        image: torch.Tensor,
        text_data_by_model: Dict[str, Dict[str, torch.Tensor]],
        non_blocking: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Always calculate loss per model and use for updating the adversarial example.
        losses_per_model: Dict[str, torch.Tensor] = {}
        handles = {}

        # TODO: How to check whether or not this is blocking?
        # Ran experiments to confirm.
        for model_idx, (model_name, model_wrapper) in enumerate(self.vlms_dict.items()):
            model_wrapper_device = model_wrapper.device

            # # Implementation #0.
            # # Compute the loss for each model
            # loss = model_wrapper.compute_loss(
            #     image=image.to(model_wrapper_device, non_blocking=non_blocking),
            #     input_ids=text_data_by_model[model_name]["input_ids"].to(
            #         model_wrapper_device,
            #     ),
            #     attention_mask=text_data_by_model[model_name]["attention_mask"].to(
            #         model_wrapper_device,
            #     ),
            #     labels=text_data_by_model[model_name]["labels"].to(
            #         model_wrapper_device,
            #     ),
            # )
            # losses_per_model[model_name] = loss.to(
            #     self.device,
            # )

            # Implementation #1.
            # Compute the loss for each model
            # loss = model_wrapper.compute_loss(
            #     image=image.to(model_wrapper_device, non_blocking=non_blocking),
            #     input_ids=text_data_by_model[model_name]["input_ids"].to(
            #         model_wrapper_device,
            #         non_blocking=non_blocking,
            #     ),
            #     attention_mask=text_data_by_model[model_name]["attention_mask"].to(
            #         model_wrapper_device,
            #         non_blocking=non_blocking,
            #     ),
            #     labels=text_data_by_model[model_name]["labels"].to(
            #         model_wrapper_device,
            #         non_blocking=non_blocking,
            #     ),
            # )
            # losses_per_model[model_name] = loss.to(
            #     self.device, non_blocking=non_blocking
            # )

            # Implementation #2.
            # image = image.to(model_wrapper_device, non_blocking=non_blocking)
            # input_ids = text_data_by_model[model_name]["input_ids"].to(
            #     model_wrapper_device,
            #     non_blocking=non_blocking,
            # )
            # attention_mask = text_data_by_model[model_name]["attention_mask"].to(
            #     model_wrapper_device,
            #     non_blocking=non_blocking,
            # )
            # labels = text_data_by_model[model_name]["labels"].to(
            #     model_wrapper_device,
            #     non_blocking=non_blocking,
            # )

            # # Fork the computation, i.e., schedule it to run in parallel
            # handles[model_name] = torch.jit.fork(
            #     model_wrapper.compute_loss,
            #     image=image,
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels,
            # )

            # Implementation #3.
            handles[model_name] = torch.jit.fork(
                model_wrapper.compute_loss,
                image=image.to(model_wrapper_device, non_blocking=non_blocking),
                input_ids=text_data_by_model[model_name]["input_ids"].to(
                    model_wrapper_device,
                    non_blocking=non_blocking,
                ),
                attention_mask=text_data_by_model[model_name]["attention_mask"].to(
                    model_wrapper_device,
                    non_blocking=non_blocking,
                ),
                labels=text_data_by_model[model_name]["labels"].to(
                    model_wrapper_device,
                    non_blocking=non_blocking,
                ),
            )

        # # Collect results from all models
        for model_name, handle in handles.items():
            loss = torch.jit.wait(handle)
            losses_per_model[model_name] = loss.to(self.device, non_blocking=True)

        # Calculate average loss
        losses_per_model["avg"] = torch.mean(
            torch.stack(list(losses_per_model.values()))
        )
        return losses_per_model

    def disable_vlm_gradients(self):
        # set all models' requires_grad to False
        for vlm_str, vlm_wrapper in self.vlms_dict.items():
            vlm_wrapper.disable_model_gradients()

    @staticmethod
    def compute_whether_generation_begins_with_target(
        model_generations: List[str],
        targets: List[str],
    ) -> float:
        avg = np.mean(
            [gen.startswith(target) for gen, target in zip(model_generations, targets)]
        )
        return avg

    # @torch.no_grad()
    # def evaluate_jailbreak_against_vlms_and_log(
    #     self,
    #     image: torch.Tensor,
    #     prompts_and_targets_dict: Dict[str, List[str]],
    #     text_dataloader: torch.utils.data.DataLoader,
    #     # harmbench_evaluator: HarmBenchEvaluator,
    #     # llamaguard_evalutor: LlamaGuardEvaluator,
    #     wandb_logging_step_idx: int = 1,
    # ) -> Dict[str, Dict[str, Any]]:
    #     total_losses_per_model = {}
    #     total_samples = 0.0
    #     image = image.to(torch.bfloat16)
    #     for batch_idx, batch_text_data_by_model in enumerate(text_dataloader):
    #         batch_size: int = batch_text_data_by_model[
    #             list(batch_text_data_by_model.keys())[
    #                 0
    #             ]  # Any key will work for obtaining batch size.
    #         ]["input_ids"].shape[0]
    #         with torch.no_grad():
    #             batch_losses_per_model = self.compute_loss(
    #                 image=image,
    #                 text_data_by_model=batch_text_data_by_model,
    #             )
    #             print("Batch losses per model: ", batch_losses_per_model)
    #             for model_name, loss in batch_losses_per_model.items():
    #                 if model_name not in total_losses_per_model:
    #                     total_losses_per_model[model_name] = batch_size * loss
    #                 else:
    #                     total_losses_per_model[model_name] += batch_size * loss
    #             total_samples += batch_size
    #
    #     total_losses_per_model = {
    #         f"eval/loss_model={model_name}": total_loss / total_samples
    #         for model_name, total_loss in total_losses_per_model.items()
    #     }
    #     print("Total losses per model: ", total_losses_per_model)
    #
    #     # Choose a fixed subset of samples to evaluate.
    #     batch_prompts, batch_targets = self.sample_prompts_and_targets(
    #         prompts=prompts_and_targets_dict["prompts"],
    #         targets=prompts_and_targets_dict["targets"],
    #         batch_size=10,
    #     )
    #
    #     evaluation_results = {}
    #     for (
    #         model_name,
    #         model_wrapper,
    #     ) in self.vlms_dict.items():
    #         batch_model_generations = model_wrapper.generate(
    #             image=image,
    #             prompts=batch_prompts,
    #         )
    #         model_adv_generation_begins_with_target = (
    #             self.compute_whether_generation_begins_with_target(
    #                 model_generations=batch_model_generations,
    #                 targets=batch_targets,
    #             )
    #         )
    #
    #         evaluation_results[model_name] = {
    #             f"generations_{model_name}_step={wandb_logging_step_idx}": wandb.Table(
    #                 columns=[
    #                     "prompt",
    #                     "generated",
    #                     "target",
    #                 ],
    #                 data=[
    #                     [
    #                         prompt,
    #                         model_generation,
    #                         target,
    #                     ]
    #                     for prompt, model_generation, target in zip(
    #                         batch_prompts,
    #                         batch_model_generations,
    #                         batch_targets,
    #                     )
    #                 ],
    #             ),
    #             "eval/generation_begins_with_target": model_adv_generation_begins_with_target,
    #             "eval/loss": total_losses_per_model[f"eval/loss_model={model_name}"],
    #         }
    #
    #     return evaluation_results

    # def sample_prompts_and_targets(
    #     self, prompts: List[str], targets: List[str], batch_size: int = 10
    # ):
    #     batch_idx = random.sample(range(len(prompts)), batch_size)
    #     batch_prompts = [
    #         prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
    #     ]
    #     batch_targets = [
    #         target for idx, target in enumerate(targets) if idx in batch_idx
    #     ]
    #
    #     return batch_prompts, batch_targets

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        non_blocking: bool = False,
    ):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if non_blocking is not None:
            kwargs["non_blocking"] = non_blocking
        for model_name, model_wrapper in self.vlms_dict.items():
            self.vlms_dict[model_name] = model_wrapper.to(**kwargs)
        return self
