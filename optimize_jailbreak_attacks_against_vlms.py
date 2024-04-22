import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str

import ast
import json
import lightning
import lightning.pytorch.callbacks
import lightning.pytorch.loggers
import math
import os
import pprint
import torch
import wandb
from typing import Any, Dict, List

# torch.use_deterministic_algorithms(True)

import src.data
from src.globals import default_attack_config
import src.systems
import src.utils


def optimize_vlm_adversarial_examples():
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak",
        config=default_attack_config,
        entity=wandb_username,
    )
    wandb_config = dict(wandb.config)

    # Log the effective batch size.
    wandb.config.update(
        {
            "effective_batch_size": wandb_config["data"]["batch_size"]
            * wandb_config["lightning_kwargs"]["accumulate_grad_batches"]
        }
    )

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

    # Compute how many epochs we need, based on accumulate gradient steps and total steps.
    n_train_epochs = math.ceil(
        wandb_config["n_grad_steps"]
        * wandb_config["lightning_kwargs"]["accumulate_grad_batches"]
        / wandb_config["prompts_and_targets_kwargs"]["n_unique_prompts_and_targets"]
        / wandb_config["lightning_kwargs"]["limit_train_batches"]
    )

    callbacks = []
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        callbacks.extend(
            [
                # DeviceStatsMonitor()
            ]
        )
        print("GPUs available: ", devices)
    else:
        accelerator = "cpu"
        devices = None
        callbacks.extend([])
        print("No GPU available.")

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = lightning.pytorch.Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=wandb_config["lightning_kwargs"][
            "accumulate_grad_batches"
        ],
        callbacks=callbacks,
        check_val_every_n_epoch=0,
        default_root_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
        # deterministic=True,
        devices=devices,
        limit_train_batches=wandb_config["lightning_kwargs"]["limit_train_batches"],
        logger=lightning.pytorch.loggers.WandbLogger(experiment=run),
        log_every_n_steps=wandb_config["lightning_kwargs"]["log_loss_every_n_steps"],
        # overfit_batches=1,  # useful for debugging
        gradient_clip_val=wandb_config["lightning_kwargs"]["gradient_clip_val"],
        # gradient_clip_val=None,  # default
        max_epochs=n_train_epochs,
        min_epochs=n_train_epochs,
        # profiler="simple",  # Simplest profiler
        # profiler="advanced",  # More advanced profiler
        precision=wandb_config["lightning_kwargs"]["precision"],
    )

    # https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
    # "Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:"
    with trainer.init_module():
        vlm_ensemble_system = src.systems.VLMEnsembleAttackingSystem(
            wandb_config=wandb_config,
        )

    tokenized_dir_path = src.data.tokenize_prompts_and_targets_using_vlm_ensemble(
        vlm_ensemble=vlm_ensemble_system.vlm_ensemble,
        prompts_and_targets_kwargs=wandb_config["prompts_and_targets_kwargs"],
        prompts_and_targets_dir="prompts_and_targets",
        split="train",
    )

    # We need to load the VLMs ensemble in order to tokenize the dataset.
    text_datamodule = src.data.VLMEnsembleTextDataModule(
        vlm_names=list(vlm_ensemble_system.vlm_ensemble.vlms_dict.keys()),
        tokenized_dir_path=tokenized_dir_path,
        wandb_config=wandb_config,
    )

    if wandb_config["compile"]:
        # vlm_ensemble_system: VLMEnsemble = torch.compile(
        #     vlm_ensemble_system,
        #     mode="default",  # Good balance between performance and overhead.
        # )
        print(
            "Reminder: torch.compile() doesn't work. Some memory leak? Need to debug."
        )

    trainer.fit(
        model=vlm_ensemble_system,
        datamodule=text_datamodule,
    )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    optimize_vlm_adversarial_examples()
