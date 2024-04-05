from accelerate import Accelerator
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data
from typing import Dict, List, Tuple
import wandb

from src.models.ensemble import VLMEnsemble
from src.image_handling import save_multi_images


class AdversarialAttacker:
    def __init__(
        self,
        vlm_ensemble: VLMEnsemble,
        accelerator: Accelerator,
        attack_kwargs: Dict[str, any],
        **kwargs,
    ):
        self.vlm_ensemble = vlm_ensemble
        self.accelerator = accelerator
        # Check that attack kwargs has the required keys.
        assert "batch_size" in attack_kwargs
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
            attack_results = self.attack(
                image=image,
                text_dataloader=text_dataloader,
                prompts_and_targets_dict=prompts_and_targets_dict,
                **kwargs,
            )
            adv_x = attack_results["adversarial_image"]
            losses_history: Dict[str, np.ndarray] = attack_results["losses_history"]
            # prob_masses_history = torch.exp(-losses_history)

            save_multi_images(adv_x, results_dir, begin_id=image_idx)

            plt.close()
            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), squeeze=False)
            for model_str in self.vlm_ensemble.vlms_dict:
                plt.plot(
                    np.arange(len(losses_history[model_str])),
                    losses_history[model_str],
                    label=model_str,
                )
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.ylim(bottom=0.0)
            plt.legend()

            wandb.log(
                {
                    "original_image": wandb.Image(image, caption="Original Image"),
                    "adversarial_image": wandb.Image(
                        adv_x, caption="Adversarial Image"
                    ),
                    "loss_curve": wandb.Image(plt),
                }
            )

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
