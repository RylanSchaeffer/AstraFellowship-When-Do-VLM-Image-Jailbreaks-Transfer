from accelerate import Accelerator
import ast
import json
import os
import pprint
import torch
import wandb
from typing import Any, Dict, List


from src.globals import default_attack_config
from src.models.ensemble import VLMEnsemble
import src.utils


def optimize_vlm_adversarial_examples():
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak",
        config=default_attack_config,
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

    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["models_to_attack"] = ast.literal_eval(
        wandb_config["models_to_attack"]
    )

    src.utils.set_seed(seed=wandb_config["seed"])

    # Load initial image plus prompt and target data.
    tensor_images: torch.Tensor = src.utils.create_initial_images(
        image_kwargs=wandb_config["image_kwargs"],
    )

    accelerator = Accelerator()

    # We need to load the VLMs ensemble in order to tokenize the dataset.
    vlm_ensemble: VLMEnsemble = src.utils.instantiate_vlm_ensemble(
        model_strs=wandb_config["models_to_attack"],
        model_generation_kwargs=wandb_config["model_generation_kwargs"],
        accelerator=accelerator,
    )

    # We need to load the VLMs ensemble in order to tokenize the dataset.
    prompts_and_targets_dict, text_dataloader = src.utils.create_text_dataloader(
        vlm_ensemble=vlm_ensemble,
        prompt_and_targets_kwargs=wandb_config["prompt_and_targets_kwargs"],
        wandb_config=wandb_config,
        split="train",
    )
    wandb.config.update(
        {"train_indices": str(prompts_and_targets_dict["indices"].tolist())},
        # allow_val_change=True,
    )

    if wandb_config["compile"]:
        vlm_ensemble: VLMEnsemble = torch.compile(
            vlm_ensemble,
            mode="default",  # good balance between performance and overhead
            # mode="reduce-overhead",  # not guaranteed to work, but good for small batches.
        )

    attacker = src.utils.create_attacker(
        wandb_config=wandb_config,
        vlm_ensemble=vlm_ensemble,
        accelerator=accelerator,
    )

    attacker.compute_adversarial_examples(
        tensor_images=tensor_images,
        text_dataloader=text_dataloader,
        prompts_and_targets_dict=prompts_and_targets_dict,
        results_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
    )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    optimize_vlm_adversarial_examples()
