from accelerate import Accelerator
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.utils.data
import tqdm
from typing import Any, Dict, List, Tuple
import wandb

from src.models.ensemble import VLMEnsemble
from src.models.evaluators import HarmBenchEvaluator, LlamaGuardEvaluator
from src.image_handling import save_multi_images


class JailbreakAttacker:
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        accelerator: Accelerator,
        attack_kwargs: Dict[str, any],
        **kwargs,
    ):
        self.vlm_ensemble = vlm_ensemble
        self.accelerator = accelerator
        self.attack_kwargs = attack_kwargs

    @abstractmethod
    def attack(
        self,
        image: torch.Tensor,
        text_dataloader: torch.utils.data.DataLoader,
        prompts_and_targets_dict: Dict[str, List[str]],
        **kwargs,
    ):
        pass

    def optimize_image_jailbreaks(
        self,
        tensor_images: torch.Tensor,
        text_dataloader: torch.utils.data.DataLoader,
        prompts_and_targets_dict: Dict[str, List[str]],
        results_dir: str,
        **kwargs,
    ):
        os.makedirs(results_dir, exist_ok=True)
        # tensor_images = self.accelerator.prepare(tensor_images)

        for image_idx, image in enumerate(tensor_images):
            self.attack(
                image=image,
                text_dataloader=text_dataloader,
                prompts_and_targets_dict=prompts_and_targets_dict,
                **kwargs,
            )

            # save_multi_images(adv_x, results_dir, begin_id=image_idx)
            print(f"Adversarial image {image_idx+1} optimized.")

    @staticmethod
    def compute_whether_generation_begins_with_target(
        model_generations: List[str],
        targets: List[str],
    ) -> float:
        avg = np.mean(
            [gen.startswith(target) for gen, target in zip(model_generations, targets)]
        )
        return avg

    def evaluate_jailbreak_against_vlms_and_log(
        self,
        vlm_ensemble: VLMEnsemble,
        image: torch.Tensor,
        prompts_and_targets_dict: Dict[str, List[str]],
        text_dataloader: torch.utils.data.DataLoader,
        # harmbench_evaluator: HarmBenchEvaluator,
        # llamaguard_evalutor: LlamaGuardEvaluator,
        wandb_logging_step_idx: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        total_losses_per_model = {}
        total_samples = 0.0
        for batch_idx, batch_text_data_by_model in enumerate(
            tqdm.tqdm(text_dataloader)
        ):
            batch_size = batch_text_data_by_model[
                list(batch_text_data_by_model.keys())[
                    0
                ]  # Any key will work for obtaining batch size.
            ]["input_ids"].shape[0]
            with torch.no_grad():
                batch_losses_per_model = vlm_ensemble.compute_loss(
                    image=image,
                    text_data_by_model=batch_text_data_by_model,
                )
                for model_name, loss in batch_losses_per_model.items():
                    if model_name not in total_losses_per_model:
                        total_losses_per_model[model_name] = batch_size * loss
                    else:
                        total_losses_per_model[model_name] += batch_size * loss
                total_samples += batch_size

        total_losses_per_model = {
            f"eval/loss_model={model_name}": total_loss / total_samples
            for model_name, total_loss in total_losses_per_model.items()
        }

        # Choose a fixed subset of samples to evaluate.
        batch_prompts, batch_targets = self.sample_prompts_and_targets(
            prompts=prompts_and_targets_dict["prompts"],
            targets=prompts_and_targets_dict["targets"],
            batch_size=10,
        )

        evaluation_results = {}
        for (
            model_name,
            model_wrapper,
        ) in vlm_ensemble.vlms_dict.items():
            batch_model_generations = model_wrapper.generate(
                image=image,
                prompts=batch_prompts,
            )
            model_adv_generation_begins_with_target = (
                self.compute_whether_generation_begins_with_target(
                    model_generations=batch_model_generations,
                    targets=batch_targets,
                )
            )

            # TODO: Add a key about which model was attacked and which model was evaluated.
            evaluation_results[model_name] = {
                f"generations_{model_name}_step={wandb_logging_step_idx}": wandb.Table(
                    columns=[
                        "prompt",
                        "generated",
                        "target",
                    ],
                    data=[
                        [
                            prompt,
                            model_generation,
                            target,
                        ]
                        for prompt, model_generation, target in zip(
                            batch_prompts,
                            batch_model_generations,
                            batch_targets,
                        )
                    ],
                ),
                # Pytorch uses (C, H, W), but wandb uses (H, W, C).
                # See https://github.com/wandb/wandb/issues/393#issuecomment-1808432690.
                # "adversarial_image": wandb.Image(
                #     image[0].detach().numpy().transpose(1, 2, 0),
                #     caption="Adversarial Image",
                # ),
                "eval/generation_begins_with_target": model_adv_generation_begins_with_target,
            }
            evaluation_results[model_name].update(total_losses_per_model)

        return evaluation_results

    def sample_prompts_and_targets(
        self, prompts: List[str], targets: List[str], batch_size: int = 10
    ):
        batch_idx = random.sample(range(len(prompts)), batch_size)
        batch_prompts = [
            prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
        ]
        batch_targets = [
            target for idx, target in enumerate(targets) if idx in batch_idx
        ]

        return batch_prompts, batch_targets
