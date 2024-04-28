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
import time
import torch
import wandb
from typing import Any, Dict, List

# torch.use_deterministic_algorithms(True)

import src.data
from src.globals import default_attack_config
import src.systems
import src.utils


def optimize_vlm_adversarial_examples():
    run = wandb.init(
        project="universal-vlm-jailbreak",
        config=default_attack_config,
        entity=src.utils.retrieve_wandb_username(),
    )
    wandb_config = dict(wandb.config)

    # Ensure that this is a float and bounded between 0 and 1.
    wandb_config["lightning_kwargs"]["limit_train_batches"] = float(
        wandb_config["lightning_kwargs"]["limit_train_batches"]
    )
    assert 0.0 < wandb_config["lightning_kwargs"]["limit_train_batches"] <= 1.0

    # Log the effective batch size.
    wandb.config.update(
        {
            "batch_size_effective": wandb_config["data"]["batch_size"]
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

    # Compute how many epochs we need, based on accumulate gradient steps and total steps and datset size.
    train_dataset_len = src.data.get_dataset_length(
        dataset=wandb_config["data"]["dataset"],
        split="train",  # Hard-code this.
    )
    n_train_epochs = 1
    print("Number of Train Epochs: ", n_train_epochs)

    callbacks = []
    accelerator = "gpu"
    devices = torch.cuda.device_count()
    callbacks.extend(
        [
            # DeviceStatsMonitor()
        ]
    )
    print("GPUs available: ", devices)
    # if torch.cuda.is_available():
        
    # else:
    #     accelerator = "cpu"
    #     devices = None
    #     callbacks.extend([])
    #     print("No GPU available.")
    #     raise ValueError("No GPU available.")

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
        strategy="fsdp",  # Fully Sharded Data Parallelism.
    )

    # https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html
    # "Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:"
    with trainer.init_module():
        vlm_ensemble_system = src.systems.VLMEnsembleAttackingSystem(
            wandb_config=wandb_config,
        )

    limit = wandb_config["prompts_and_targets_kwargs"]["n_unique_prompts_and_targets"]
    prompts_and_targets = src.data.load_prompts_and_targets_v2(
        dataset=wandb_config["data"]["dataset"],
        split="train",  # Hard-code this.
    )[:limit]
    batch_size= wandb_config["data"]["batch_size"]
    print("Batch Size: ", batch_size)
    # We need to load the VLMs ensemble in order to tokenize the dataset.
    text_datamodule = src.data.TextDataModule(
        ensemble=vlm_ensemble_system.vlm_ensemble,
        prompts_and_targets=prompts_and_targets,
        batch_size=batch_size,
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

    # Convert to float32 for generation.
    vlm_ensemble_system.tensor_image = vlm_ensemble_system.tensor_image.to(
        torch.float32
    )
    n_generations = 100

    wandb.log(
                {
                    f"jailbreak_image_step={vlm_ensemble_system.optimizer_step_counter}": wandb.Image(
                        # https://docs.wandb.ai/ref/python/data-types/image
                        # 0 removes the size-1 batch dimension.
                        # The transformation doesn't accept bfloat16.
                        data_or_path=vlm_ensemble_system.convert_tensor_to_pil_image(
                            vlm_ensemble_system.tensor_image[0]
                        ),
                        # caption="Adversarial Image",
                    ),
                },
            )
    # Load prompts for generation spot-checking.
    for split in ["train", "eval"]:
        prompts_and_targets_dict = src.data.load_prompts_and_targets(
            dataset=wandb_config["data"]["dataset"],
            split=split,
        )

        for model_name_str in vlm_ensemble_system.vlm_ensemble.vlms_dict:
            # We need the tensor dtypes to agree.
            vlm_ensemble_system.vlm_ensemble.vlms_dict[
                model_name_str
            ].model.llm_backbone.llm = vlm_ensemble_system.vlm_ensemble.vlms_dict[
                model_name_str
            ].model.llm_backbone.llm.to(
                torch.float32
            )
            model_generations_dict = {
                "generations": [],
                "prompts": [],
                "targets": [],
            }
            for prompt_idx, (prompt, target) in enumerate(
                zip(
                    prompts_and_targets_dict["prompts"][
                        : n_generations
                    ],
                    prompts_and_targets_dict["targets"][
                        : n_generations
                    ],
                )
            ):
                start_time = time.time()
                model_generations = vlm_ensemble_system.vlm_ensemble.vlms_dict[
                    model_name_str
                ].generate(image=vlm_ensemble_system.tensor_image, prompts=[prompt])
                model_generations_dict["generations"].extend(model_generations)
                model_generations_dict["prompts"].extend([prompt])
                model_generations_dict["targets"].extend([target])
                end_time = time.time()
                print(
                    f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {model_generations[0]}\nGeneration Duration: {end_time - start_time} seconds\n\n"
                )

            wandb_log_data = {
                f"generations_model={model_name_str}_split={split}_optimizer_step_counter={vlm_ensemble_system.optimizer_step_counter}": wandb.Table(
                    columns=[
                        "prompt",
                        "generated",
                        "target",
                    ],
                    data=[
                        [
                            prompt,
                            model_generation,
                            target,
                        ]
                        for prompt, model_generation, target in zip(
                            model_generations_dict["prompts"],
                            model_generations_dict["generations"],
                            model_generations_dict["targets"],
                        )
                    ],
                ),
                "optimizer_step_counter": vlm_ensemble_system.optimizer_step_counter,
            }
            wandb.log(wandb_log_data)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    optimize_vlm_adversarial_examples()
