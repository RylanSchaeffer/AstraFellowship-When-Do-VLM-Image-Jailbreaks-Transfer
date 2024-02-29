# Adapted from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py#L20
import matplotlib.pyplot as plt
import random

import numpy as np
import seaborn as sns
import torch
import tqdm
from typing import Dict, List
import wandb

from src.attacks.base import AdversarialAttacker
from src.image_handling import normalize_images


class PGDAttacker(AdversarialAttacker):
    def __init__(
        self,
        models_to_attack_dict: Dict[str, torch.nn.Module],
        models_to_eval_dict: Dict[str, torch.nn.Module],
        attack_kwargs: Dict[str, any],
    ):
        assert "step_size" in attack_kwargs
        super(PGDAttacker, self).__init__(
            models_to_attack_dict=models_to_attack_dict,
            models_to_eval_dict=models_to_eval_dict,
            attack_kwargs=attack_kwargs,
        )

        self.losses_history: np.ndarray = np.zeros(1)

    def attack(
        self,
        image: torch.Tensor,
        prompts: List[str],
        targets: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ensure loss history is empty.
        self.losses_history = np.zeros(
            shape=(
                self.attack_kwargs["total_steps"] + 1,
                len(self.models_to_attack_dict),
            )
        )

        assert len(prompts) == len(targets)
        dataset_size = len(prompts)

        # TODO: Tokenize the prompts once.
        # prompts_per_model = {
        #     model_str: prompts for model_str, model_wrapper in self.models_to_attack_dict.items()
        # }
        #
        # prompt = prompt_wrapper.LlavaLlama2Prompt(
        #     self.model, self.tokenizer, text_prompts=text_prompt, device=self.device
        # )

        # Ensure gradients are computed for the image.
        original_image = image.clone()
        image.requires_grad_(True)
        image.retain_grad()

        for step_idx in tqdm.tqdm(range(self.attack_kwargs["total_steps"] + 1)):
            batch_idx = random.sample(
                range(dataset_size), self.attack_kwargs["batch_size"]
            )
            batch_prompts = [
                prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
            ]
            batch_targets = [
                target for idx, target in enumerate(targets) if idx in batch_idx
            ]

            # Occasionally generate to see what the VLM is outputting.
            if (step_idx % self.attack_kwargs["generate_every_n_steps"]) == 0:
                for model_name, model_wrapper in self.models_to_eval_dict.items():
                    batch_model_generations = model_wrapper.generate(
                        image=image,
                        prompts=batch_prompts,
                    )
                    model_generation_begins_with_target = np.mean(
                        [
                            gen.startswith(target)
                            for gen, target in zip(
                                batch_model_generations, batch_targets
                            )
                        ]
                    )
                    wandb.log(
                        {
                            f"generations_{model_name}_step={step_idx}": wandb.Table(
                                columns=[
                                    "prompt",
                                    "generated",
                                    "target text",
                                ],
                                data=[
                                    [prompt, model_generation, target]
                                    for prompt, model_generation, target in zip(
                                        batch_prompts,
                                        batch_model_generations,
                                        batch_targets,
                                    )
                                ],
                            ),
                            # Pytorch expected c, h, w, but wandb expects h, w, c.
                            # See https://github.com/wandb/wandb/issues/393#issuecomment-1808432690.
                            "adversarial_image": wandb.Image(
                                image[0].detach().numpy().transpose(1, 2, 0),
                                caption="Adversarial Image",
                            ),
                            "model_generation_begins_with_target": model_generation_begins_with_target,
                        },
                        step=step_idx + 1,
                    )

            # Always calculate loss per model and use for updating the adversarial example.
            target_loss_per_model: Dict[str, torch.Tensor] = {}
            total_target_loss = torch.zeros(1, requires_grad=True, device="cpu")
            for model_idx, (model_name, model_wrapper) in enumerate(
                self.models_to_attack_dict.items()
            ):
                target_loss_for_model = model_wrapper.compute_loss(
                    image=image,
                    prompts=batch_prompts,
                    targets=batch_targets,
                )
                target_loss_per_model[model_name] = target_loss_for_model.item()
                self.losses_history[step_idx, model_idx] = target_loss_for_model.item()
                total_target_loss = total_target_loss + target_loss_for_model.cpu()
            total_target_loss = total_target_loss / len(self.models_to_attack_dict)
            target_loss_per_model["avg"] = total_target_loss.item()
            total_target_loss.backward()

            image.data = (
                image.data
                - self.attack_kwargs["step_size"] * image.grad.detach().sign()
            ).clamp(0.0, 1.0)
            image.grad.zero_()

            wandb.log(
                target_loss_per_model,
                step=step_idx + 1,
            )

        attack_results = {
            "original_image": original_image,
            "adversarial_image": image,
            "losses_history": self.losses_history.copy(),
        }

        return attack_results
