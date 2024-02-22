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
from src.image_handling import get_list_image
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

    models_to_eval_dict = src.utils.instantiate_models(
        model_strs=wandb_config["models_to_eval"],
        prompts=prompts,
        targets=targets,
        split="eval",
    )

    # Slice the models to attack.
    models_to_attack_dict = {
        model_str: model
        for model_str, model in models_to_eval_dict.items()
        if model_str in wandb_config["models_to_attack"]
    }

    attacker = src.utils.create_attacker(
        wandb_config=wandb_config,
        models_to_attack_dict=models_to_attack_dict,
        models_to_eval_dict=models_to_eval_dict,
    )

    attacker.compute_adversarial_examples(
        images=images,
        prompts=prompts,
        targets=targets,
        results_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
    )


if __name__ == "__main__":
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak",
        config=default_config,
        entity=wandb_username,
    )
    wandb_config = dict(wandb.config)

    # Create checkpoint directory for this run, and save the config to the directory.
    wandb_run_dir = os.path.join("runs", wandb.run.id)
    os.makedirs(wandb_run_dir)
    wandb_config["wandb_run_dir"] = wandb_run_dir
    with open(os.path.join(wandb_run_dir, "wandb_config.json"), "w") as fp:
        json.dump(obj=wandb_config, fp=fp)

    pp = pprint.PrettyPrinter(indent=4)
    print("W&B Config:")
    pp.pprint(wandb_config)

    # These are both sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["models_to_attack"] = ast.literal_eval(
        wandb_config["models_to_attack"]
    )
    wandb_config["models_to_eval"] = ast.literal_eval(wandb_config["models_to_eval"])
    # Ensure that the attacked models are also evaluated.
    wandb_config["models_to_eval"].update(wandb_config["models_to_attack"])

    generate_vlm_adversarial_examples(wandb_config=wandb_config)
