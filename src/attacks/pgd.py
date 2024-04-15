# Adapted from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py#L20
from accelerate import Accelerator
import math
import matplotlib.pyplot as plt
import numpy as np
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

    def optimize_image_jailbreak(
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
            elif self.attack_kwargs["precision"] == "float64":
                dtype = torch.float64
            else:
                raise NotImplementedError
            adv_image = adv_image.to(dtype=dtype)
        adv_image.requires_grad_(True)
        adv_image.retain_grad()

        n_train_epochs = math.ceil(
            self.attack_kwargs["total_steps"] / len(text_dataloader)
        )

        gradient_step = 0
        for epoch_idx in range(n_train_epochs):
            for batch_idx, batch_text_data_by_model in enumerate(text_dataloader):
                if (gradient_step % self.attack_kwargs["log_image_every_n_steps"]) == 0:
                    wandb_logging_step_idx = gradient_step + 1
                    wandb.log(
                        {
                            f"jailbreak_image_step={gradient_step}": wandb.Image(
                                # https://docs.wandb.ai/ref/python/data-types/image
                                data_or_path=self.convert_tensor_to_pil_image(
                                    adv_image[0].to(
                                        torch.float32
                                    )  # The transformation doesn't accept bfloat16.
                                ),
                                caption="Adversarial Image",
                            ),
                        },
                        step=wandb_logging_step_idx,
                    )

                if (gradient_step % self.attack_kwargs["log_loss_every_n_steps"]) == 0:
                    wandb_logging_step_idx = gradient_step + 1
                    with torch.no_grad():
                        # Note: This has type torch.float32.
                        uint8_adv_image = (255.0 * adv_image).to(
                            dtype=torch.uint8
                        ) / 255.0
                        # Log some numerical values to later debug.
                        print(
                            f"uint8 image at step {gradient_step}:\n",
                            uint8_adv_image[0],
                        )
                        uint8_losses_per_model = self.vlm_ensemble.compute_loss(
                            image=uint8_adv_image,
                            text_data_by_model=batch_text_data_by_model,
                        )

                    wandb.log(
                        {
                            f"train_uint8/loss_{key}": value.item()
                            for key, value in uint8_losses_per_model.items()
                        },
                        step=wandb_logging_step_idx,
                    )

                losses_per_model = self.vlm_ensemble.compute_loss(
                    image=adv_image,
                    text_data_by_model=batch_text_data_by_model,
                )

                if (gradient_step % self.attack_kwargs["log_loss_every_n_steps"]) == 0:
                    wandb_logging_step_idx = gradient_step + 1

                    # Log some numerical values to later debug.
                    print(f"full image at step {gradient_step}:\n", adv_image[0])

                    # Log the losses to W&B.
                    wandb.log(
                        {
                            f"train/loss_{key}": value.item()
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

        attack_results = {
            "original_image": original_image,
            "adversarial_image": adv_image,
        }

        return attack_results
