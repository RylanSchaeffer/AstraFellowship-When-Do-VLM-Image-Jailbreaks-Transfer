import os
import time

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
import pandas as pd

import src.data
import src.globals
from src.models.evaluators import HarmBenchEvaluator, LlamaGuardEvaluator
import src.systems
import src.utils


def evaluate_vlm_adversarial_examples():
    wandb_username = "chuajamessh"
    run = wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=src.globals.default_eval_config,
        entity=src.utils.retrieve_wandb_username(),
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

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["model_to_eval"] = ast.literal_eval(wandb_config["model_to_eval"])

    src.utils.set_seed(seed=wandb_config["seed"])

    callbacks = [
        # src.systems.VLMEnsembleEvaluatingCallback(
        #     wandb_config=wandb_config,
        # )
    ]
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
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
        wandb_run_id=wandb_config["wandb_run_id"],
        wandb_sweep_id=None,
        # refresh=True,
        refresh=False,
    )
    # print("Number of Jailbreak Images: ", len(runs_jailbreak_dict_list)

    # # Rylan uses this for debugging.
    # runs_jailbreak_dict_list = runs_jailbreak_dict_list[-1:]

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
    # tokenized_dir_path = src.data.tokenize_prompts_and_targets_using_vlm_ensemble(
    #     vlm_ensemble=vlm_ensemble_system.vlm_ensemble,
    #     data_kwargs=wandb_config["data"],
    #     split=wandb_config["data"]["split"],
    # )

    # text_datamodule = src.data.VLMEnsembleTextDataModule(
    #     vlm_names=list(vlm_ensemble_system.vlm_ensemble.vlms_dict.keys()),
    #     tokenized_dir_path=tokenized_dir_path,
    #     wandb_config=wandb_config,
    # )

    # Load the raw prompts to use for generate.
    prompts_and_targets_dict = src.data.load_prompts_and_targets(
        dataset=wandb_config["data"]["dataset"],
        split=wandb_config["data"]["split"],
    )

    models_to_eval= list(wandb_config["model_to_eval"])

    # model: str, 
    to_log: list[dict] =[]
    for model in models_to_eval:
        print(f"Eval Model: {model}")
        for run_jailbreak_dict in runs_jailbreak_dict_list:
            # Read image from disk. This image data should match the uint8 images.
            # Shape: Batch-Channel-Height-Width
            adv_image = (
                torchvision.transforms.v2.functional.pil_to_tensor(
                    Image.open(run_jailbreak_dict["file_path"], mode="r")
                ).unsqueeze(0)
                / 255.0
            )

            wandb_additional_data = {
                "eval_model_str": model,
                "wandb_run_id": run_jailbreak_dict["wandb_run_id"],
                "optimizer_step_counter": run_jailbreak_dict["optimizer_step_counter"],
                "attack_models_str": run_jailbreak_dict["attack_models_str"],
            }

            # Bind data to the evaluation system for W&B logging on epoch end.
            vlm_ensemble_system.tensor_image.data = adv_image
            vlm_ensemble_system.wandb_additional_data = wandb_additional_data

            # Compute the loss.
            # trainer.test(
            #     model=vlm_ensemble_system,
            #     datamodule=text_datamodule,
            # )
            # generate one token 

            model_generations_dict = {
                "generations": [],
                "prompts": [],
                "targets": [],
                "matches_target": [],
            }
            # # Move to the CPU for faster sampling.
            # # Will explicitly placing on CPU cause issues?
            # vlm_ensemble_system.vlm_ensemble = vlm_ensemble_system.vlm_ensemble.to("cpu")
            for prompt_idx, (prompt, target) in enumerate(
                zip(
                    prompts_and_targets_dict["prompts"][: wandb_config["n_generations"]],
                    prompts_and_targets_dict["targets"][: wandb_config["n_generations"]],
                )
            ):
                start_time = time.time()
                model_generations = vlm_ensemble_system.vlm_ensemble.vlms_dict[
                    model
                ].generate(image=adv_image, prompts=[prompt])
                assert len(model_generations) == 1, "wait what??"
                single_gen = model_generations[0]
                matches_target = single_gen.strip().lower() == target.strip().lower()
                model_generations_dict["generations"].append(single_gen)
                model_generations_dict["prompts"].append(prompt)
                model_generations_dict["targets"].append(target)
                model_generations_dict["matches_target"].append(matches_target)
                end_time = time.time()
                if prompt_idx % 10 == 0:
                    print(
                        f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {model_generations[0]}\nGeneration Duration: {end_time - start_time} seconds\n\n"
                    )
            model_generations_dict["success_rate"] = np.mean(model_generations_dict["matches_target"])

            merged_dict = {**wandb_additional_data, **model_generations_dict}
            # run_jailbreak_dict["generations_prompts_targets_evals"] = model_generations_dict

            to_log.append(merged_dict)
            wandb.log(merged_dict)
    # make a pd dataframe
    df = pd.DataFrame(to_log)
    wandb.log({"evaluations": wandb.Table(dataframe=df)})
    # for data in to_log:


    # # Delete the VLM because we no longer need it and we want to reclaim the memory for
    # # the evaluation VLM.
    # del vlm_ensemble_system.vlm_ensemble
    # del vlm_ensemble_system
    # for evaluator_model_name_str, eval_llm_constr in [
    #     ("LlamaGuard2", LlamaGuardEvaluator),
    #     ("HarmBench", HarmBenchEvaluator),
    # ]:
    #     eval_llm = eval_llm_constr()
    #     for run_jailbreak_dict in runs_jailbreak_dict_list:
    #         run_jailbreak_dict["generations_prompts_targets_evals"][
    #             f"model_eval_{evaluator_model_name_str}"
    #         ] = [
    #             eval_llm.evaluate(prompt=prompt, generation=generation)
    #             for prompt, generation in zip(
    #                 run_jailbreak_dict["generations_prompts_targets_evals"]["prompts"],
    #                 run_jailbreak_dict["generations_prompts_targets_evals"][
    #                     "generations"
    #                 ],
    #             )
    #         ]
    #         run_jailbreak_dict[
    #             f"score_model={evaluator_model_name_str}"
    #         ] = eval_llm.compute_score(
    #             judgements=run_jailbreak_dict["generations_prompts_targets_evals"][
    #                 f"model_eval_{evaluator_model_name_str}"
    #             ],
    #         )
    #
    # for run_jailbreak_dict in runs_jailbreak_dict_list:
    #     generations_prompts_targets_evals_dict = run_jailbreak_dict[
    #         "generations_prompts_targets_evals"
    #     ]
    #     wandb_log_data = {
    #         f"generations_{model_name_str}_optimizer_step={run_jailbreak_dict['optimizer_step_counter']}": wandb.Table(
    #             columns=[
    #                 "prompt",
    #                 "generated",
    #                 "target",
    #                 "LlamaGuard2",
    #                 "HarmBench",
    #             ],
    #             data=[
    #                 [
    #                     prompt,
    #                     model_generation,
    #                     target,
    #                     llama_guard2_eval,
    #                     harmbench_eval,
    #                 ]
    #                 for prompt, model_generation, target, llama_guard2_eval, harmbench_eval in zip(
    #                     generations_prompts_targets_evals_dict["prompts"],
    #                     generations_prompts_targets_evals_dict["generations"],
    #                     generations_prompts_targets_evals_dict["targets"],
    #                     generations_prompts_targets_evals_dict[
    #                         f"model_eval_LlamaGuard2"
    #                     ],
    #                     generations_prompts_targets_evals_dict[f"model_eval_HarmBench"],
    #                 )
    #             ],
    #         ),
    #         "eval_model_str": model_name_str,
    #         "wandb_run_id": run_jailbreak_dict["wandb_run_id"],
    #         "optimizer_step_counter": run_jailbreak_dict["optimizer_step_counter"],
    #         "attack_models_str": run_jailbreak_dict["attack_models_str"],
    #         "loss/score_model=LlamaGuard2": run_jailbreak_dict[
    #             "score_model=LlamaGuard2"
    #         ],
    #         "loss/score_model=HarmBench": run_jailbreak_dict["score_model=HarmBench"],
    #     }
    #     wandb.log(wandb_log_data)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    evaluate_vlm_adversarial_examples()
