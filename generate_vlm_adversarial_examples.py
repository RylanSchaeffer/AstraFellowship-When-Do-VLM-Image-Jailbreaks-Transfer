import ast
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import pprint
from tqdm import tqdm
from torchvision import transforms
import wandb
from typing import Any

from src.attacks import SSA_CommonWeakness
from src.globals import default_config
from src.image_handling import save_multi_images, get_list_image
import src.utils


def generate_vlm_adversarial_examples(wandb_config: dict[str, Any]):
    src.utils.set_seed(seed=wandb_config["seed"])

    if wandb_config["image_initialization"] == "NIPS17":
        images = get_list_image("src/dataset/NIPS17")
        resizer = transforms.Resize((224, 224))
        images = [resizer(i).unsqueeze(0) for i in images]
    # elif wandb_config["image_initialization"] == "random":
    #     images = torch.rand((1, 3, 224, 224))
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                wandb_config["image_initialization"]
            )
        )

    class GPT4AttackCriterion:
        def __init__(self):
            self.count = 0

        def __call__(self, loss, *args):
            self.count += 1
            if self.count % 120 == 0:
                print(loss)
            return -loss

    prompts, targets = src.utils.load_prompts_and_targets(
        prompts_and_targets_str=wandb_config["prompts_and_targets"]
    )

    wandb.log(
        {
            "text": wandb.Table(
                columns=["prompt text", "target text"],
                data=[
                    [prompt, target] for prompt, target in zip(prompts, targets)
                ],  # TODO: Generalize this to many.
            )
        }
    )

    models_list = []
    for model_str in wandb_config["models_to_attack"]:
        if model_str == "blip2":
            from src.models.blip2 import Blip2VisionModel

            models_list.append(
                Blip2VisionModel(prompt=prompt_text, target_text=target_text)
            )
        elif model_str == "instruct_blip":
            from src.models.instruct_blip import InstructBlipVisionModel

            models_list.append(
                InstructBlipVisionModel(prompt=prompt_text, target_text=target_text)
            )
        elif model_str == "gpt4":
            from src.models.minigpt4 import get_gpt4_image_model

            models_list.append(get_gpt4_image_model(target_text=target_text))
        else:
            raise ValueError("Invalid model_str: {}".format(model_str))

    if wandb_config["attack_kwargs"]["attack_name"] == "ssa_common_weakness":
        attacker = SSA_CommonWeakness(
            models_list=models_list,
            epsilon=wandb_config["attack_kwargs"]["epsilon"],
            step_size=wandb_config["attack_kwargs"]["step_size"],
            total_step=wandb_config["attack_kwargs"]["total_steps"],
            criterion=GPT4AttackCriterion(),
        )
    else:
        raise ValueError(
            "Invalid attack_name: {}".format(
                wandb_config["attack_kwargs"]["attack_name"]
            )
        )

    id = 0
    attacks_dir = os.path.join(wandb_config["wandb_run_dir"], "attacks")
    os.makedirs(attacks_dir, exist_ok=True)
    for i, x in enumerate(tqdm(images)):
        if i >= 200:
            break
        x = x.cuda()
        adv_x, losses_history = attacker(x, None)
        save_multi_images(adv_x, attacks_dir, begin_id=id)
        id += x.shape[0]

        plt.close()
        for model_idx in range(len(models_list)):
            plt.plot(
                list(range(len(losses_history))),
                losses_history[:, model_idx],
                label=wandb_config["models_to_attack"][model_idx],
            )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()

        wandb.log(
            {
                "original_image": wandb.Image(x, caption="Original Image"),
                "adversarial_image": wandb.Image(adv_x, caption="Adversarial Image"),
                "loss_curve": wandb.Image(plt),
            }
        )


if __name__ == "__main__":
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak",
        config=default_config,
        entity=wandb_username,
    )
    wandb_config = dict(wandb.config)

    wandb_config["models_to_attack"] = ast.literal_eval(
        wandb_config["models_to_attack"]
    )
    wandb_config["models_to_test"] = ast.literal_eval(wandb_config["models_to_test"])

    # Create checkpoint directory for this run, and save the config to the directory.
    wandb_run_dir = os.path.join("runs", wandb.run.id)
    os.makedirs(wandb_run_dir)
    wandb_config["wandb_run_dir"] = wandb_run_dir
    with open(os.path.join(wandb_run_dir, "wandb_config.json"), "w") as fp:
        json.dump(obj=wandb_config, fp=fp)

    pp = pprint.PrettyPrinter(indent=4)
    print("W&B Config:")
    pp.pprint(wandb_config)
    generate_vlm_adversarial_examples(wandb_config=wandb_config)
