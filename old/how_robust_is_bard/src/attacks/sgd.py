"""
PGD: Projected Gradient Descent
"""
import torch
from torch import nn
from typing import Callable, Dict, List, Tuple
import wandb

from .base import AdversarialImageAttacker


class SGDAttack(AdversarialImageAttacker):
    def __init__(
        self,
        models_to_attack_dict: Dict[str, nn.Module],
        total_steps: int = 10,
        random_start: bool = False,
        step_size: float = 0.1,
        criterion: Callable = nn.CrossEntropyLoss(),
        targeted_attack=False,
        generate_every_n_steps: int = 1000,
        *args,
        **kwargs,
    ):
        self.random_start = random_start
        self.total_steps = total_steps
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.generate_every_n_steps = generate_every_n_steps
        super(SGDAttack, self).__init__(
            models_to_attack_dict=models_to_attack_dict, *args, **kwargs
        )

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(
        self,
        image: torch.Tensor,
        prompts: List[str],
        targets: List[str],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_image = image.clone()
        if self.random_start:
            image = self.perturb(image)

        losses_history = torch.full(
            size=(self.total_steps, len(self.models_to_attack_dict)),
            fill_value=float("inf"),
        )
        for step_idx in range(self.total_steps):
            image.requires_grad = True
            loss = 0.0
            for model_idx, (model_str, model) in enumerate(
                self.models_to_attack_dict.items()
            ):
                model_loss = model.compute_loss(
                    image=image,
                    prompts=prompts,
                    targets=targets,
                ).to(image.device)
                losses_history[step_idx, model_idx] = model_loss.item()
                loss += model_loss

                if step_idx % self.generate_every_n_steps == 0:
                    generations = model.generate(image, prompts)
                    wandb.log(
                        {
                            f"generations_{model_str}_step={step_idx}": wandb.Table(
                                columns=[
                                    "prompt text",
                                    "generated text",
                                    "target text",
                                ],
                                data=[
                                    [prompt, generation, target]
                                    for prompt, generation, target in zip(
                                        prompts, generations, targets
                                    )
                                ],
                            )
                        },
                    )

            loss.backward()
            with torch.no_grad():
                grad = image.grad
                # image.requires_grad = False
                image = image - self.step_size * grad
                image = self.clamp(image, original_image)

            wandb.log(
                {
                    f"loss_{model_str}": losses_history[step_idx, model_idx]
                    for model_idx, model_str in enumerate(self.models_to_attack_dict)
                },
            )
            print(f"Step {step_idx + 1}/{self.total_steps}: {loss.item()}")

        return image, losses_history
