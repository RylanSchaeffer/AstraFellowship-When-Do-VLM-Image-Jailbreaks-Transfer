from accelerate import Accelerator
import ast
import json
import numpy as np
import os
import pprint
import wandb
from typing import Any, List


from src.globals import default_config
import src.utils


def generate_vlm_adversarial_examples():
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

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["models_to_attack"] = ast.literal_eval(
        wandb_config["models_to_attack"]
    )
    wandb_config["models_to_eval"] = ast.literal_eval(wandb_config["models_to_eval"])
    # Ensure that the attacked models are also evaluated.
    wandb_config["models_to_eval"].update(wandb_config["models_to_attack"])

    assert all(
        model_str in wandb_config["model_generation_kwargs"]
        for model_str in wandb_config["models_to_eval"]
    )

    src.utils.set_seed(seed=wandb_config["seed"])

    # Load data.
    tensor_images_list = src.utils.create_or_load_images(
        image_kwargs=wandb_config["image_kwargs"],
    )
    prompts_and_targets_by_split = src.utils.load_prompts_and_targets(
        prompts_and_targets_kwargs=wandb_config["prompt_and_targets_kwargs"],
    )

    accelerator = Accelerator()

    vlm_ensemble = src.utils.instantiate_vlm_ensemble(
        models_to_attack=wandb_config["models_to_attack"],
        models_to_eval=wandb_config["models_to_eval"],
        model_generation_kwargs=wandb_config["model_generation_kwargs"],
        accelerator=accelerator,
    )

    attacker = src.utils.create_attacker(
        wandb_config=wandb_config,
        vlm_ensemble=vlm_ensemble,
        accelerator=accelerator,
    )

    attacker.compute_adversarial_examples(
        tensor_images_list=tensor_images_list,
        prompts_and_targets_by_split=prompts_and_targets_by_split,
        results_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
    )


if __name__ == "__main__":
    generate_vlm_adversarial_examples()
