from accelerate import Accelerator
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
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

    def compute_adversarial_examples(
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
        adv_image: torch.Tensor,
        prompts_and_targets_dict: Dict[str, List[str]],
        text_dataloader: torch.utils.data.DataLoader,
        harmbench_evaluator: HarmBenchEvaluator,
        llamaguard_evalutor: LlamaGuardEvaluator,
        wandb_logging_step_idx: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        total_losses_per_model = {}
        for batch_idx, batch_text_data_by_model in enumerate(
            tqdm.tqdm(text_dataloader)
        ):
            with torch.no_grad():
                batch_losses_per_model = vlm_ensemble.compute_loss(
                    image=adv_image,
                    text_data_by_model=batch_text_data_by_model,
                )

        batch_prompts, batch_targets = self.sample_prompts_and_targets(
            prompts=prompts_and_targets_dict["test"]["prompts"],
            targets=prompts_and_targets_dict["test"]["targets"],
        )

        evaluation_results = {}
        for (
            model_name,
            model_wrapper,
        ) in vlm_ensemble.vlms_dict.items():
            batch_model_generations = model_wrapper.generate(
                image=adv_image,
                prompts=batch_prompts,
            )
            model_adv_generation_begins_with_target = (
                self.compute_whether_generation_begins_with_target(
                    model_generations=batch_model_generations,
                    targets=batch_targets,
                )
            )

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
                "adversarial_image": wandb.Image(
                    adv_image[0].detach().numpy().transpose(1, 2, 0),
                    caption="Adversarial Image",
                ),
                "test/generation_begins_with_target": model_adv_generation_begins_with_target,
            }

        return evaluation_results
