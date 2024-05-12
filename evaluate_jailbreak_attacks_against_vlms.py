import copy
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
import gc
import lightning
import numpy as np
from PIL import Image
import pprint
import time
import torch
import torchvision.transforms.v2.functional
import wandb
from typing import Any, Dict, List


import src.data
import src.globals
import src.models.evaluators
import src.systems
import src.utils


def evaluate_vlm_adversarial_examples():
    run = wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=src.globals.default_eval_config,
        entity=src.utils.retrieve_wandb_username(),
    )
    wandb_config: Dict[str, Any] = dict(wandb.config)

    # Ensure that this is a float and bounded between 0 and 1.
    wandb_config["lightning_kwargs"]["limit_eval_batches"] = float(
        wandb_config["lightning_kwargs"]["limit_eval_batches"]
    )
    assert 0.0 < wandb_config["lightning_kwargs"]["limit_eval_batches"] <= 1.0

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
    assert torch.cuda.device_count() == 1

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["model_to_eval"] = ast.literal_eval(wandb_config["model_to_eval"])

    src.utils.set_seed(seed=wandb_config["seed"])

    callbacks = []
    if torch.cuda.is_available():
        accelerator = "gpu"
        # Need to set this to 1 otherwise Lightning will try DDP or FSDP and fuck things up.
        devices = 1
        callbacks.extend(
            [
                # DeviceStatsMonitor()
            ]
        )
    else:
        accelerator = "cpu"
        devices = "auto"
        callbacks.extend([])
        print("No GPU available.")

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = lightning.pytorch.Trainer(
        accelerator=accelerator,
        callbacks=callbacks,
        check_val_every_n_epoch=0,
        default_root_dir=os.path.join(wandb_config["wandb_run_dir"], "results"),
        # deterministic=True,
        devices=devices,
        limit_test_batches=wandb_config["lightning_kwargs"]["limit_eval_batches"],
        logger=lightning.pytorch.loggers.WandbLogger(experiment=run),
        log_every_n_steps=wandb_config["lightning_kwargs"]["log_loss_every_n_steps"],
        # overfit_batches=1,  # useful for debugging
        max_epochs=1,
        min_epochs=1,
        # profiler="simple",  # Simplest profiler
        # profiler="advanced",  # More advanced profiler
        precision=wandb_config["lightning_kwargs"]["precision"],
    )

    # Load jailbreak images' paths.
    runs_jailbreak_dict_list = src.utils.load_jailbreak_dicts_list(
        wandb_attack_run_id=wandb_config["wandb_attack_run_id"],
        wandb_sweep_id=None,
        # refresh=True,
        refresh=False,
    )
    wandb.config.update(
        {
            "models_to_attack": runs_jailbreak_dict_list[0]["models_to_attack"],
        }
    )

    # Rylan uses this for debugging.
    # runs_jailbreak_dict_list = runs_jailbreak_dict_list[-2:]

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

    # Ensure that the tokenized dataset exists.
    tokenized_dir_path = src.data.tokenize_prompts_and_targets_using_vlm_ensemble(
        vlm_ensemble=vlm_ensemble_system.vlm_ensemble,
        data_kwargs=wandb_config["data"],
        split=wandb_config["data"]["split"],
    )

    text_datamodule = src.data.VLMEnsembleTextDataModule(
        vlm_names=list(vlm_ensemble_system.vlm_ensemble.vlms_dict.keys()),
        tokenized_dir_path=tokenized_dir_path,
        wandb_config=wandb_config,
    )

    # Load the raw prompts to use for generate.
    prompts_and_targets_dict = src.data.load_prompts_and_targets(
        dataset=wandb_config["data"]["dataset"],
        split=wandb_config["data"]["split"],
    )
    if wandb_config["n_generations"] == "None":
        num_prompts = len(prompts_and_targets_dict["prompts"])
        wandb.config.update({"n_generations": num_prompts})
        wandb_config["n_generations"] = num_prompts

    # There should only be one.
    model_name_str = list(wandb_config["model_to_eval"])[0]
    for jailbreak_idx, run_jailbreak_dict in enumerate(runs_jailbreak_dict_list):
        # Read image from disk. This image data should match the uint8 images.
        # Shape: Batch-Channel-Height-Width
        adv_image = (
            torchvision.transforms.v2.functional.pil_to_tensor(
                Image.open(run_jailbreak_dict["file_path"], mode="r")
            ).unsqueeze(0)
            / 255.0
        )

        wandb_additional_data = {
            "eval_model_str": model_name_str,
            "wandb_attack_run_id": run_jailbreak_dict["wandb_attack_run_id"],
            "optimizer_step_counter": run_jailbreak_dict["optimizer_step_counter"],
            "models_to_attack": run_jailbreak_dict["models_to_attack"],
        }

        # Bind data to the evaluation system for W&B logging on epoch end.
        vlm_ensemble_system.tensor_image.data = adv_image
        vlm_ensemble_system.wandb_additional_data = wandb_additional_data

        # Compute the loss.
        trainer.test(
            model=vlm_ensemble_system,
            datamodule=text_datamodule,
        )

        # Only generate every 1000 optimizer steps.
        if (run_jailbreak_dict["optimizer_step_counter"] % 1000) != 0:
            continue

        generations_prompts_targets_evals_dict = {
            "prompts": [],
            "generations": [],
            "targets": [],
        }
        for prompt_idx, (prompt, target) in enumerate(
            zip(
                prompts_and_targets_dict["prompts"][: wandb_config["n_generations"]],
                prompts_and_targets_dict["targets"][: wandb_config["n_generations"]],
            )
        ):
            start_time = time.time()
            # TODO: Add a batch dimension.
            generation = vlm_ensemble_system.vlm_ensemble.vlms_dict[
                model_name_str
            ].generate(image=adv_image, prompts=[prompt])[0]
            generations_prompts_targets_evals_dict["generations"].extend([generation])
            generations_prompts_targets_evals_dict["prompts"].extend([prompt])
            generations_prompts_targets_evals_dict["targets"].extend([target])
            end_time = time.time()
            # print(
            #     f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {generation}\nGeneration Duration: {end_time - start_time} seconds\n\n"
            # )

        run_jailbreak_dict[
            "generations_prompts_targets_evals_dict"
        ] = generations_prompts_targets_evals_dict
        runs_jailbreak_dict_list[jailbreak_idx] = copy.deepcopy(run_jailbreak_dict)

    # Free up memory.
    del vlm_ensemble_system
    del text_datamodule
    del trainer
    gc.collect()

    # Score generations using LlamaGuard2 and HarmBench.
    for eval_model_name, eval_model_constr in [
        ("claude3opus", src.models.evaluators.Claude3OpusEvaluator),
        ("harmbench", src.models.evaluators.HarmBenchEvaluator),
        ("llamaguard2", src.models.evaluators.LlamaGuard2Evaluator),
    ]:
        eval_model = eval_model_constr()
        for jailbreak_idx, run_jailbreak_dict in enumerate(runs_jailbreak_dict_list):
            if "generations_prompts_targets_evals_dict" not in run_jailbreak_dict:
                continue

            run_jailbreak_dict["generations_prompts_targets_evals_dict"][
                f"judgements_{eval_model_name}"
            ] = eval_model.evaluate(
                prompts=run_jailbreak_dict["generations_prompts_targets_evals_dict"][
                    "prompts"
                ],
                generations=run_jailbreak_dict[
                    "generations_prompts_targets_evals_dict"
                ]["generations"],
            )
            run_jailbreak_dict[
                f"loss/score_{eval_model_name}"
            ] = eval_model.compute_score(
                run_jailbreak_dict["generations_prompts_targets_evals_dict"][
                    f"judgements_{eval_model_name}"
                ]
            )
            runs_jailbreak_dict_list[jailbreak_idx] = copy.deepcopy(run_jailbreak_dict)

        # Clean up the eval model once we're done with it.
        del eval_model
        gc.collect()

    for run_jailbreak_dict in runs_jailbreak_dict_list:
        if "generations_prompts_targets_evals_dict" not in run_jailbreak_dict:
            continue

        generations_prompts_targets_evals_dict = run_jailbreak_dict[
            "generations_prompts_targets_evals_dict"
        ]
        wandb_log_data = {
            f"generations_{model_name_str}_optimizer_step={run_jailbreak_dict['optimizer_step_counter']}": wandb.Table(
                columns=[
                    "prompt",
                    "generated",
                    "target",
                    "llamaguard2",
                    "harmbench",
                    "claude3opus",
                ],
                data=[
                    [
                        prompt,
                        model_generation,
                        target,
                        llama_guard2_eval,
                        harmbench_eval,
                        claude3opus_eval,
                    ]
                    for prompt, model_generation, target, llama_guard2_eval, harmbench_eval, claude3opus_eval in zip(
                        generations_prompts_targets_evals_dict["prompts"],
                        generations_prompts_targets_evals_dict["generations"],
                        generations_prompts_targets_evals_dict["targets"],
                        generations_prompts_targets_evals_dict[
                            "judgements_llamaguard2"
                        ],
                        generations_prompts_targets_evals_dict["judgements_harmbench"],
                        generations_prompts_targets_evals_dict[
                            "judgements_claude3opus"
                        ],
                    )
                ],
            ),
            "eval_model_str": model_name_str,
            "wandb_attack_run_id": run_jailbreak_dict["wandb_attack_run_id"],
            "optimizer_step_counter": run_jailbreak_dict["optimizer_step_counter"],
            "optimizer_step_counter_epoch": run_jailbreak_dict[
                "optimizer_step_counter"
            ],
            "models_to_attack": run_jailbreak_dict["models_to_attack"],
            "loss/score_model=llamaguard2": run_jailbreak_dict[
                "loss/score_llamaguard2"
            ],
            "loss/score_model=harmbench": run_jailbreak_dict["loss/score_harmbench"],
            "loss/score_model=claude3opus": run_jailbreak_dict[
                "loss/score_claude3opus"
            ],
        }
        wandb.log(wandb_log_data)

    wandb.finish()


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    evaluate_vlm_adversarial_examples()
