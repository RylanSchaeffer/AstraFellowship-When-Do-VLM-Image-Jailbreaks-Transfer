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
        super(PGDAttacker, self).__init__(
            models_to_attack_dict=models_to_attack_dict,
            models_to_eval_dict=models_to_eval_dict,
            attack_kwargs=attack_kwargs,
        )

        self.loss_history: np.ndarray = np.zeros(1)

    def attack(
        self,
        image: torch.Tensor,
        prompts: List[str],
        targets: List[str],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # Ensure loss history is empty.
        self.loss_history = np.zeros(
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
        image = image.half()
        image = image.cuda()
        image.requires_grad_(True)
        image.retain_grad()

        for step_idx in tqdm.tqdm(range(self.attack_kwargs["total_steps"] + 1)):
            batch_idx = random.sample(
                range(dataset_size), self.attack_kwargs["batch_size"]
            )

            # Occasionally generate to see what the VLM is outputting.
            if (step_idx % self.attack_kwargs["generate_every_n_steps"]) == 0:
                for model_name, model_wrapper in self.models_to_eval_dict.items():
                    model_generations = model_wrapper.generate(
                        image=image,
                        prompts=prompts,
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
                                        prompts, model_generations, targets
                                    )
                                ],
                            )
                        },
                    )

            # Always calculate loss per model and use for updating the adversarial example.
            target_loss_per_model: Dict[str, torch.Tensor] = {}
            for model_name, model_wrapper in self.models_to_attack_dict.items():
                target_loss_for_model = model_wrapper.compute_loss(
                    image=image,
                    prompts=[
                        prompt for idx, prompt in enumerate(prompts) if idx in batch_idx
                    ],
                    targets=[
                        target for idx, target in enumerate(targets) if idx in batch_idx
                    ],
                )
                target_loss_per_model[model_name] = target_loss_for_model
            total_target_loss = torch.mean(target_loss_per_model.values())
            target_loss_per_model["total"] = total_target_loss
            total_target_loss.backward()

            image.data = (
                image.data
                - self.attack_kwargs["step_size"] * image.grad.detach().sign()
            ).clamp(0.0, 1.0)
            image.grad.zero_()

            self.loss_history.append(total_target_loss.item())

            wandb.log(
                {k: v.item() for k, v in target_loss_per_model.items()},
                step=step_idx + 1,
            )

            # if step_idx % 100 == 0:
            #     print("######### Output - Iter = %d ##########" % step_idx)
            #     x_adv = normalize(image)
            #     response = my_generator.generate(prompt, x_adv)
            #     print(">>>", response)
            #
            #     adv_img_prompt = denormalize(x_adv).detach().cpu()
            #     adv_img_prompt = adv_img_prompt.squeeze(0)
            #     save_image(
            #         adv_img_prompt,
            #         "%s/bad_prompt_temp_%d.bmp"
            #         % (self.wandb_config.save_dir, step_idx),
            #     )

        return original_image, image

    def plot_loss(self):
        plt.close()
        for model_idx, model_str in enumerate(self.models_to_attack_dict):
            plt.plot(
                list(range(len(self.losses_history))),
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
                "adversarial_image": wandb.Image(adv_x, caption="Adversarial Image"),
                "loss_curve": wandb.Image(plt),
            }
        )

    # def _compute_loss(
    #     self, image: torch.Tensor, prompts: List[str], targets: List[str]
    # ):
    #
    #     to_regress_tokens = [
    #         torch.as_tensor([item[1:]]).cuda()
    #         for item in self.tokenizer(targets).input_ids
    #     ]  # get rid of the default <bos> in targets tokenization.
    #
    #     seq_tokens_length = []
    #     labels = []
    #     input_ids = []
    #
    #     for i, item in enumerate(to_regress_tokens):
    #         L = item.shape[1] + context_length[i]
    #         seq_tokens_length.append(L)
    #
    #         context_mask = torch.full(
    #             [1, context_length[i]],
    #             -100,
    #             dtype=to_regress_tokens[0].dtype,
    #             device=to_regress_tokens[0].device,
    #         )
    #         labels.append(torch.cat([context_mask, item], dim=1))
    #         input_ids.append(torch.cat([context_input_ids[i], item], dim=1))
    #
    #     # padding token
    #     pad = torch.full(
    #         [1, 1],
    #         0,
    #         dtype=to_regress_tokens[0].dtype,
    #         device=to_regress_tokens[0].device,
    #     ).cuda()  # it does not matter ... Anyway will be masked out from attention...
    #
    #     max_length = max(seq_tokens_length)
    #     attention_mask = []
    #
    #     for i in range(batch_size):
    #         # padding to align the length
    #         num_to_pad = max_length - seq_tokens_length[i]
    #
    #         padding_mask = torch.full(
    #             [1, num_to_pad], -100, dtype=torch.long, device=self.device
    #         )
    #         labels[i] = torch.cat([labels[i], padding_mask], dim=1)
    #
    #         input_ids[i] = torch.cat([input_ids[i], pad.repeat(1, num_to_pad)], dim=1)
    #         attention_mask.append(
    #             torch.LongTensor([[1] * (seq_tokens_length[i]) + [0] * num_to_pad])
    #         )
    #
    #     labels = torch.cat(labels, dim=0).cuda()
    #     input_ids = torch.cat(input_ids, dim=0).cuda()
    #     attention_mask = torch.cat(attention_mask, dim=0).cuda()
    #
    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         return_dict=True,
    #         labels=labels,
    #         images=images,
    #     )
    #     loss = outputs.loss
    #
    #     return loss
