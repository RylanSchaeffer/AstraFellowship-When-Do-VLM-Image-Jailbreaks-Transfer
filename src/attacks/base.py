import os
from abc import abstractmethod
from math import ceil
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from typing import Dict, List
import wandb

from src.image_handling import save_multi_images


class AdversarialAttacker:
    def __init__(
        self,
        models_to_attack_dict: Dict[str, torch.nn.Module],
        models_to_eval_dict: Dict[str, torch.nn.Module],
        attack_kwargs: Dict[str, any],
        **kwargs,
    ):
        self.models_to_attack_dict = models_to_attack_dict
        self.models_to_eval_dict = models_to_eval_dict
        # Check that attack kwargs has the required keys.
        assert "batch_size" in attack_kwargs

        self.attack_kwargs = attack_kwargs
        self.disable_model_gradients()
        self.distribute_models()
        self.device = torch.device("cuda")
        self.n = len(self.models_to_attack_dict)

    @abstractmethod
    def attack(
        self, image: torch.Tensor, prompts: List[str], targets: List[str], **kwargs
    ):
        pass

    def compute_adversarial_examples(
        self,
        images: List[torch.Tensor],
        prompts: List[str],
        targets: List[str],
        results_dir: str,
        **kwargs,
    ):
        os.makedirs(results_dir, exist_ok=True)

        for image_idx, image in enumerate(images):
            attack_results = self.attack(
                image=image, prompts=prompts, targets=targets, **kwargs
            )
            adv_x = attack_results["adversarial_image"]
            losses_history = attack_results["losses_history"]
            # TODO: Compute probability masses for loss history.
            # prob_masses_history = torch.exp(-losses_history)

            save_multi_images(adv_x, results_dir, begin_id=image_idx)

            plt.close()
            for model_idx, model_str in enumerate(self.models_to_attack_dict):
                plt.plot(
                    list(range(len(losses_history))),
                    losses_history[:, model_idx],
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

    def disable_model_gradients(self):
        # set the model parameters requires_grad is False
        for wrapper_model in self.models_to_eval_dict.values():
            wrapper_model.model.requires_grad_(False)
            wrapper_model.model.eval()

    def distribute_models(self):
        """
        make each model on one gpu
        :return:
        """
        num_gpus = torch.cuda.device_count()
        models_each_gpu = ceil(len(self.models_to_eval_dict) / num_gpus)
        for i, wrapper_model in enumerate(self.models_to_eval_dict.values()):
            wrapper_model.model.to(
                torch.device(f"cuda:{num_gpus - 1 - i // models_each_gpu}")
            )
            wrapper_model.device = torch.device(
                f"cuda:{num_gpus - 1 - i // models_each_gpu}"
            )

    def to(self, device: torch.device):
        for wrapper_model in self.models_to_attack_dict.values():
            wrapper_model.model.to(device)
            wrapper_model.model.device = device
        self.device = device
