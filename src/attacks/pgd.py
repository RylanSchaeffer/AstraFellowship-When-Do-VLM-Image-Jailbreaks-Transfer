# Adapted from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py#L20
from accelerate import Accelerator
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch
import tqdm
from typing import Dict, List, Optional
import wandb

from src.attacks.base import AdversarialAttacker
from src.models.ensemble import VLMEnsemble
from src.image_handling import normalize_images


class PGDAttacker(AdversarialAttacker):
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        accelerator: Accelerator,
        attack_kwargs: Dict[str, any],
    ):
        assert "step_size" in attack_kwargs
        super(PGDAttacker, self).__init__(
            accelerator=accelerator,
            attack_kwargs=attack_kwargs,
            vlm_ensemble=vlm_ensemble,
        )

        self.losses_history: Optional[np.ndarray] = None

    def attack(
        self,
        image: torch.Tensor,
        prompts_and_targets_by_split: Dict[str, Dict[str, List[str]]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ensure loss history is empty.
        self.reinitialize_losses_history()

        # Ensure gradients are computed for the image.
        original_image = image.clone()
        adv_image = image
        adv_image.requires_grad_(True)
        adv_image.retain_grad()

        for step_idx in tqdm.tqdm(range(self.attack_kwargs["total_steps"] + 1)):
            # Occasionally generate to see what the VLM is outputting.
            if (step_idx % self.attack_kwargs["test_every_n_steps"]) == 0:
                self.test_attack_against_vlms_and_log(
                    original_image=original_image,
                    adv_image=adv_image,
                    prompts_and_targets_by_split=prompts_and_targets_by_split,
                    step_idx=step_idx,
                )

            batch_prompts, batch_targets = self.sample_prompts_and_targets(
                prompts=prompts_and_targets_by_split["train"]["prompts"],
                targets=prompts_and_targets_by_split["train"]["targets"],
            )
            losses_per_model = self.vlm_ensemble.compute_loss(
                image=adv_image,
                prompts=batch_prompts,
                targets=batch_targets,
            )
            # Record the losses per model.
            for key, loss in losses_per_model.items():
                self.losses_history[key][step_idx] = loss.item()

            wandb.log(
                {f"train/loss_{key}": value for key, value in losses_per_model.items()},
                step=step_idx + 1,
            )

            losses_per_model["avg"].backward()

            adv_image.data = (
                adv_image.data
                - self.attack_kwargs["step_size"] * adv_image.grad.detach().sign()
            ).clamp(0.0, 1.0)
            adv_image.grad.zero_()

        attack_results = {
            "original_image": original_image,
            "adversarial_image": adv_image,
            "losses_history": self.losses_history.copy(),
        }

        return attack_results

    def test_attack_against_vlms_and_log(
        self,
        original_image: torch.Tensor,
        adv_image: torch.Tensor,
        prompts_and_targets_by_split: Dict[str, Dict[str, List[str]]],
        step_idx: int,
    ):
        batch_prompts, batch_targets = self.sample_prompts_and_targets(
            prompts=prompts_and_targets_by_split["test"]["prompts"],
            targets=prompts_and_targets_by_split["test"]["targets"],
        )

        for (
            model_name,
            model_wrapper,
        ) in self.vlm_ensemble.vlms_to_eval_dict.items():
            batch_nonadv_model_generations = model_wrapper.generate(
                image=original_image.to(self.accelerator.device),
                prompts=batch_prompts,
            )
            model_nonadv_generation_begins_with_target = (
                self.compute_whether_generation_begins_with_target(
                    model_generations=batch_nonadv_model_generations,
                    targets=batch_targets,
                )
            )
            batch_adv_model_generations = model_wrapper.generate(
                image=adv_image,
                prompts=batch_prompts,
            )
            model_adv_generation_begins_with_target = (
                self.compute_whether_generation_begins_with_target(
                    model_generations=batch_adv_model_generations,
                    targets=batch_targets,
                )
            )

            wandb.log(
                {
                    f"generations_{model_name}_step={step_idx}": wandb.Table(
                        columns=[
                            "prompt",
                            "generated",
                            "target",
                            "adv_generated",
                        ],
                        data=[
                            [
                                prompt,
                                nonadv_model_generation,
                                target,
                                adv_model_generation,
                            ]
                            for prompt, nonadv_model_generation, adv_model_generation, target in zip(
                                batch_prompts,
                                batch_nonadv_model_generations,
                                batch_adv_model_generations,
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
                    "test/model_nonadv_generation_begins_with_target": model_nonadv_generation_begins_with_target,
                    "test/model_adv_generation_begins_with_target": model_adv_generation_begins_with_target,
                },
                step=step_idx + 1,
            )

            with torch.no_grad():
                losses_per_model = self.vlm_ensemble.compute_loss(
                    image=adv_image,
                    prompts=batch_prompts,
                    targets=batch_targets,
                )
                wandb.log(
                    {f"test/loss_{key}": value for key, value in losses_per_model.items()},
                    step=step_idx + 1,
                )

    def reinitialize_losses_history(self):
        self.losses_history = {
            model_str: np.full(self.attack_kwargs["total_steps"] + 1, fill_value=np.nan)
            for model_str in self.vlm_ensemble.vlms_to_eval_dict
        }
        self.losses_history["avg"] = np.full(
            self.attack_kwargs["total_steps"] + 1, fill_value=np.nan
        )

    def sample_prompts_and_targets(self, prompts: List[str], targets: List[str]):
        batch_idx = random.sample(range(len(prompts)), self.attack_kwargs["batch_size"])
        batch_prompts = [
            prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
        ]
        batch_targets = [
            target for idx, target in enumerate(targets) if idx in batch_idx
        ]

        return batch_prompts, batch_targets
