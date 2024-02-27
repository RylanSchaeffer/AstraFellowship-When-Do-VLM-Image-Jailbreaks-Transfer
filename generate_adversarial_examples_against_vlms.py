import ast
import json
import os
import pprint
import torch
from torchvision import transforms
import wandb
from typing import Any, List

from src.globals import default_config
from src.image_handling import get_list_image
import src.utils


def generate_vlm_adversarial_examples(wandb_config: dict[str, Any]):
    src.utils.set_seed(seed=wandb_config["seed"])

    if wandb_config["image_initialization"] == "NIPS17":
        images_list = get_list_image("old/how_robust_is_bard/src/dataset/NIPS17")
        resizer = transforms.Resize((224, 224))
        images_list = [resizer(i).unsqueeze(0).to(torch.float16) for i in images_list]
        # Only use one image for one attack.
        images_list: List[torch.Tensor] = [images_list[wandb_config["datum_index"]]]
    elif wandb_config["image_initialization"] == "random":
        images_list: List[torch.Tensor] = [torch.rand((1, 3, 512, 512))]
    else:
        raise ValueError(
            "Invalid image_initialization: {}".format(
                wandb_config["image_initialization"]
            )
        )

    prompts, targets = src.utils.load_prompts_and_targets(
        prompts_and_targets_str=wandb_config["prompts_and_targets"]
    )
    prompts = prompts[: wandb_config["n_prompts_and_targets"]]
    targets = targets[: wandb_config["n_prompts_and_targets"]]

    models_to_eval_dict = src.utils.instantiate_models(
        model_strs=wandb_config["models_to_eval"],
        model_generation_kwargs=wandb_config["model_generation_kwargs"],
        split="eval",
    )

    # Subset the models that we will attack.
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
        images=images_list,
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

    generate_vlm_adversarial_examples(wandb_config=wandb_config)
