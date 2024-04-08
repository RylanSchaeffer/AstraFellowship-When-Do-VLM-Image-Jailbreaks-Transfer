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

from src.attacks.base import JailbreakAttacker
from src.models.ensemble import VLMEnsemble
from src.image_handling import normalize_images


class PGDAttacker(JailbreakAttacker):
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
        adv_image = image.clone()
        if "precision" in self.attack_kwargs:
            if self.attack_kwargs["precision"] == "bfloat16":
                dtype = torch.bfloat16
            elif self.attack_kwargs["precision"] == "float16":
                dtype = torch.float16
            elif self.attack_kwargs["precision"] == "float32":
                dtype = torch.float32
            else:
                raise NotImplementedError
            adv_image = adv_image.to(dtype=dtype)
        adv_image.requires_grad_(True)
        adv_image.retain_grad()

        n_train_epochs = math.ceil(
            self.attack_kwargs["total_steps"] / len(text_dataloader)
        )
        n_train_steps = n_train_epochs * len(text_dataloader)

        gradient_step = 0
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
                if (gradient_step % self.attack_kwargs["log_image_every_n_steps"]) == 0:
                    wandb_logging_step_idx = gradient_step + 1
                    wandb.log(
                        {
                            f"jailbreak_image_step={gradient_step}": wandb.Image(
                                data_or_path=adv_image,
                                caption="Adversarial Image",
                            ),
                        },
                        step=wandb_logging_step_idx,
                    )

                losses_per_model = self.vlm_ensemble.compute_loss(
                    image=adv_image,
                    text_data_by_model=batch_text_data_by_model,
                )

                if (gradient_step % self.attack_kwargs["log_loss_every_n_steps"]) == 0:
                    wandb_logging_step_idx = gradient_step + 1
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

                gradient_step += 1

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

    def sample_prompts_and_targets(self, prompts: List[str], targets: List[str]):
        batch_idx = random.sample(range(len(prompts)), self.attack_kwargs["batch_size"])
        batch_prompts = [
            prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
        ]
        batch_targets = [
            target for idx, target in enumerate(targets) if idx in batch_idx
        ]

        return batch_prompts, batch_targets
