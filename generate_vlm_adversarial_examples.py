import ast
import json
import matplotlib.pyplot as plt
import os
import pprint
import torch
from tqdm import tqdm
from torchvision import transforms
import wandb
from typing import Any

from src.globals import default_config
from src.image_handling import save_multi_images, get_list_image
import src.utils


def generate_vlm_adversarial_examples(wandb_config: dict[str, Any]):
    src.utils.set_seed(seed=wandb_config["seed"])

    if wandb_config["image_initialization"] == "NIPS17":
        images = get_list_image("src/dataset/NIPS17")
        resizer = transforms.Resize((224, 224))
        images = [resizer(i).unsqueeze(0).to(torch.float16) for i in images]
    # elif wandb_config["image_initialization"] == "random":
    #     images = torch.rand((1, 3, 224, 224))
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                wandb_config["image_initialization"]
            )
        )

    prompts, targets = src.utils.load_prompts_and_targets(
        prompts_and_targets_str=wandb_config["prompts_and_targets"]
    )
    prompts = prompts[: wandb_config["n_samples"]]
    targets = targets[: wandb_config["n_samples"]]

    wandb.log(
        {
            "text": wandb.Table(
                columns=["prompt text", "target text"],
                data=[[prompt, target] for prompt, target in zip(prompts, targets)],
            )
        }
    )

    models_to_attack_list = src.utils.instantiate_models(
        model_strs=wandb_config["models_to_attack"],
        prompts=prompts,
        targets=targets,
        split="train",
    )
    # models_to_eval_list = src.utils.instantiate_models(
    #     model_strs=wandb_config["models_to_eval"],
    #     prompts=prompts,
    #     targets=targets,
    #     split="eval",
    # )

    attacker = src.utils.create_attacker(
        wandb_config=wandb_config, models_list=models_to_attack_list
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
        for model_idx in range(len(models_to_attack_list)):
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
