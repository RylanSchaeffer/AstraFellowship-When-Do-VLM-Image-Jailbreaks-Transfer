import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str

import torchvision.transforms.v2.functional
import ast
import json
import lightning
import numpy as np
from PIL import Image
import pprint
import torch
import wandb
from typing import Any, Dict, List


import src.data
import src.globals
import src.systems
import src.utils


def evaluate_vlm_adversarial_examples():
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=src.globals.default_eval_config,
        entity=wandb_username,
    )
    wandb_config: Dict[str, Any] = dict(wandb.config)

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
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # assert len(cuda_visible_devices) == 3

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["model_to_eval"] = ast.literal_eval(wandb_config["model_to_eval"])

    src.utils.set_seed(seed=wandb_config["seed"])

    callbacks = [
        src.systems.VLMEnsembleEvaluatingCallback(
            wandb_config=wandb_config,
        )
    ]
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        callbacks.extend(
            [
                # DeviceStatsMonitor()
            ]
        )
        # os.environ["RANK"] = "0"
        # os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
    else:
        accelerator = "cpu"
        devices = None
        callbacks.extend([])
        print("No GPU available.")

    print("devices: ", devices)

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = lightning.pytorch.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        check_val_every_n_epoch=0,
        default_root_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
        # deterministic=True,
        devices=devices,
        logger=lightning.pytorch.loggers.WandbLogger(experiment=run),
        log_every_n_steps=wandb_config["lightning_kwargs"]["log_loss_every_n_steps"],
        # overfit_batches=1,  # useful for debugging
        max_epochs=1,
        min_epochs=1,
        # profiler="simple",  # Simplest profiler
        # profiler="advanced",  # More advanced profiler
        precision=wandb_config["lightning_kwargs"]["precision"],
    )

    # Load jailbreak images, their paths.
    runs_jailbreak_dict_list = src.utils.load_jailbreak_dicts_list(
        wandb_run_id=wandb_config["wandb_run_id"],
        wandb_sweep_id=None,
        # refresh=True,
        refresh=False,
    )

    # runs_jailbreak_dict_list = [
    #     ele for ele in runs_jailbreak_dict_list if ele["wandb_run_id"] == "8attseox"
    # ]
    # runs_jailbreak_dict_list = [runs_jailbreak_dict_list[-1]]

    # We need to create a placeholder image to initialize the VLMEnsembleEvaluatingSystem.
    # This ensures that Lightning can recognize the parameter and place it on the appropriate device(s).
    placeholder_adv_image = (
        torchvision.transforms.v2.functional.pil_to_tensor(
            Image.open(runs_jailbreak_dict_list[0]["file_path"], mode="r")
        ).unsqueeze(0)
        / 255.0
    )

    # https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
    # "Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:"
    with trainer.init_module():
        vlm_ensemble_system = src.systems.VLMEnsembleEvaluatingSystem(
            wandb_config=wandb_config,
            tensor_image=placeholder_adv_image,
        )

    tokenized_dir_path = src.data.tokenize_prompts_and_targets_using_vlm_ensemble(
        vlm_ensemble=vlm_ensemble_system.vlm_ensemble,
        prompts_and_targets_kwargs=wandb_config["prompts_and_targets_kwargs"],
        prompts_and_targets_dir="prompts_and_targets",
        split="eval",
    )

    # We need to load the VLMs ensemble in order to tokenize the dataset.
    text_datamodule = src.data.VLMEnsembleTextDataModule(
        vlm_names=list(vlm_ensemble_system.vlm_ensemble.vlms_dict.keys()),
        tokenized_dir_path=tokenized_dir_path,
        wandb_config=wandb_config,
    )

    model_name_str = list(wandb_config["model_to_eval"])[0]
    for run_jailbreak_dict in runs_jailbreak_dict_list:
        # Read image from disk. This image data should match the uint8 images.
        # Shape: Batch-Channel-Height-Width
        adv_image = (
            torchvision.transforms.v2.functional.pil_to_tensor(
                Image.open(run_jailbreak_dict["file_path"], mode="r")
            ).unsqueeze(0)
            / 255.0
        )

        # Bind data to the evaluation system for W&B logging on epoch end.
        vlm_ensemble_system.tensor_image.data = adv_image
        vlm_ensemble_system.wandb_additional_data = {
            "eval_model_str": model_name_str,
            "wandb_run_id": run_jailbreak_dict["wandb_run_id"],
            "optimizer_step_counter": run_jailbreak_dict["optimizer_step_counter"],
            "attack_models_str": run_jailbreak_dict["attack_models_str"],
        }

        trainer.test(
            model=vlm_ensemble_system,
            datamodule=text_datamodule,
        )

        print(
            f"Evaluated {model_name_str} on {run_jailbreak_dict['wandb_run_id']} at step {run_jailbreak_dict['optimizer_step_counter']}"
        )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    evaluate_vlm_adversarial_examples()
