# Adapted from https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py#L20
import random
import torch
from torch import Tensor
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

        self.loss_history: List[float] = []

    def attack(
        self,
        image: torch.Tensor,
        prompts: List[str],
        targets: List[str],
        **kwargs,
    ):
        # Ensure loss history is empty.
        self.loss_history = []

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
        image = image.cuda()
        image.requires_grad_(True)
        image.retain_grad()

        for step_idx in tqdm.tqdm(range(self.attack_kwargs["total_steps"] + 1)):
            batch_idx = random.sample(
                range(dataset_size), self.attack_kwargs["batch_size"]
            )

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

        return x_adv

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_history)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_history, label="Target Loss")

        # Add in a title and axes labels
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Loss")

        # Display the plot
        plt.legend(loc="best")
        plt.savefig("%s/loss_curve.png" % (self.wandb_config.save_dir))
        plt.clf()

        torch.save(self.loss_history, "%s/loss" % (self.wandb_config.save_dir))

    def _compute_loss(
        self, image: torch.Tensor, prompts: List[str], targets: List[str]
    ):
        context_length = prompts.context_length
        context_input_ids = prompts.input_ids
        batch_size = len(targets)

        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size

        to_regress_tokens = [
            torch.as_tensor([item[1:]]).cuda()
            for item in self.tokenizer(targets).input_ids
        ]  # get rid of the default <bos> in targets tokenization.

        seq_tokens_length = []
        labels = []
        input_ids = []

        for i, item in enumerate(to_regress_tokens):
            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)

            context_mask = torch.full(
                [1, context_length[i]],
                -100,
                dtype=to_regress_tokens[0].dtype,
                device=to_regress_tokens[0].device,
            )
            labels.append(torch.cat([context_mask, item], dim=1))
            input_ids.append(torch.cat([context_input_ids[i], item], dim=1))

        # padding token
        pad = torch.full(
            [1, 1],
            0,
            dtype=to_regress_tokens[0].dtype,
            device=to_regress_tokens[0].device,
        ).cuda()  # it does not matter ... Anyway will be masked out from attention...

        max_length = max(seq_tokens_length)
        attention_mask = []

        for i in range(batch_size):
            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]

            padding_mask = torch.full(
                [1, num_to_pad], -100, dtype=torch.long, device=self.device
            )
            labels[i] = torch.cat([labels[i], padding_mask], dim=1)

            input_ids[i] = torch.cat([input_ids[i], pad.repeat(1, num_to_pad)], dim=1)
            attention_mask.append(
                torch.LongTensor([[1] * (seq_tokens_length[i]) + [0] * num_to_pad])
            )

        labels = torch.cat(labels, dim=0).cuda()
        input_ids = torch.cat(input_ids, dim=0).cuda()
        attention_mask = torch.cat(attention_mask, dim=0).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
            images=images,
        )
        loss = outputs.loss

        return loss
