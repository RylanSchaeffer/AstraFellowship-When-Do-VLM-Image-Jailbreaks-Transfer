# Adapted from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py#L20
from accelerate import Accelerator
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import torch
import torch.utils.data
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
        text_dataloader: torch.utils.data.DataLoader,
        prompts_and_targets_dict: Dict[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ensure gradients are computed for the image.
        original_image = image.clone()
        adv_image = image
        adv_image.requires_grad_(True)
        adv_image.retain_grad()

        n_train_epochs = math.ceil(
            self.attack_kwargs["total_steps"] / len(text_dataloader)
        )
        n_train_steps = n_train_epochs * len(text_dataloader)

        wandb_logging_step_idx = 1
        # self.test_attack_against_vlms_and_log(
        #     original_image=original_image,
        #     adv_image=adv_image,
        #     prompts_and_targets_dict=prompts_and_targets_dict,
        #     text_dataloader=text_dataloader,
        #     wandb_logging_step_idx=wandb_logging_step_idx,
        # )

        for epoch_idx in range(n_train_epochs):
            for batch_idx, batch_text_data_by_model in enumerate(
                tqdm.tqdm(text_dataloader)
            ):
                losses_per_model = self.vlm_ensemble.compute_loss(
                    image=adv_image,
                    text_data_by_model=batch_text_data_by_model,
                )

                # Log the losses to W&B.
                wandb.log(
                    {
                        f"train/loss_{key}": value
                        for key, value in losses_per_model.items()
                    },
                    step=wandb_logging_step_idx,
                )

                losses_per_model["avg"].backward()
                adv_image.data = (
                    adv_image.data
                    - self.attack_kwargs["step_size"] * adv_image.grad.detach().sign()
                ).clamp(0.0, 1.0)
                adv_image.grad.zero_()
                wandb_logging_step_idx += 1

            # self.test_attack_against_vlms_and_log(
            #     original_image=original_image,
            #     adv_image=adv_image,
            #     prompts_and_targets_dict=prompts_and_targets_dict,
            #     wandb_logging_step_idx=wandb_logging_step_idx,
            # )

        attack_results = {
            "original_image": original_image,
            "adversarial_image": adv_image,
        }

        return attack_results

    def test_attack_against_vlms_and_log(
        self,
        original_image: torch.Tensor,
        adv_image: torch.Tensor,
        prompts_and_targets_dict: Dict[str, List[str]],
        text_dataloader: torch.utils.data.DataLoader,
        wandb_logging_step_idx: int,
    ):
        # losses_all_per_model = DefaultDict[str, List[float]]()
        for batch_idx, batch_text_data_by_model in enumerate(
            tqdm.tqdm(text_dataloader)
        ):
            with torch.no_grad():
                losses_per_model = self.vlm_ensemble.compute_loss(
                    image=adv_image,
                    text_data_by_model=batch_text_data_by_model,
                )

        wandb.log(
            {f"test/loss_{key}": value for key, value in losses_per_model.items()},
            step=wandb_logging_step_idx + 1,
        )

        batch_prompts, batch_targets = self.sample_prompts_and_targets(
            prompts=prompts_and_targets_dict["test"]["prompts"],
            targets=prompts_and_targets_dict["test"]["targets"],
        )

        for (
            model_name,
            model_wrapper,
        ) in self.vlm_ensemble.vlms_dict.items():
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
                    f"generations_{model_name}_step={wandb_logging_step_idx}": wandb.Table(
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
                step=wandb_logging_step_idx + 1,
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
