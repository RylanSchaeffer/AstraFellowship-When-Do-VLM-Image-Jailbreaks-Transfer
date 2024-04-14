import torchvision.transforms.v2.functional
from accelerate import Accelerator
import ast
import json
import numpy as np
import os
from PIL import Image
import pprint
import torch
import wandb
from typing import Any, Dict, List


from src.globals import default_eval_config
from src.models.ensemble import VLMEnsemble
from src.models.evaluators import HarmBenchEvaluator, LlamaGuardEvaluator
import src.utils


def evaluate_vlm_adversarial_examples():
    wandb_username = src.utils.retrieve_wandb_username()
    run = wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=default_eval_config,
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
    wandb_config["models_to_eval"] = ast.literal_eval(wandb_config["models_to_eval"])

    src.utils.set_seed(seed=wandb_config["seed"])

    # Load jailbreak images, their paths.
    runs_jailbreak_dict_list = src.utils.load_jailbreak_dicts_list(
        wandb_sweep_id=wandb_config["wandb_sweep_id"],
        refresh=False,
    )

    accelerator = Accelerator()
    # harmbench_evaluator = HarmBenchEvaluator(
    #     device=len(cuda_visible_devices) - 1,  # Place on the last GPU (arbitrary).
    # )
    # llamaguard_evalutor = LlamaGuardEvaluator(
    #     device=len(cuda_visible_devices) - 2,  # Place on 2nd-to-last GPU (arbitrary).
    # )

    runs_jailbreak_dict_list = [
        ele for ele in runs_jailbreak_dict_list if ele["wandb_run_id"] == "qy7ptwbj"
    ]
    runs_jailbreak_dict_list = [runs_jailbreak_dict_list[-1]]

    for model_name_str in wandb_config["models_to_eval"]:
        vlm_ensemble: VLMEnsemble = src.utils.instantiate_vlm_ensemble(
            model_strs=[model_name_str],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            accelerator=accelerator,
        )

        assert wandb_config["attack_kwargs"]["attack_name"] == "eval"
        attacker = src.utils.create_attacker(
            wandb_config=wandb_config,
            vlm_ensemble=vlm_ensemble,
            accelerator=accelerator,
        )

        # if wandb_config["compile"]:
        #     vlm_ensemble: VLMEnsemble = torch.compile(
        #         vlm_ensemble,
        #         mode="default",  # good balance between performance and overhead
        #         # mode="reduce-overhead",  # not guaranteed to work, but good for small batches.
        #     )

        for run_jailbreak_dict in runs_jailbreak_dict_list:
            # We need to load the VLMs ensemble in order to tokenize the dataset.
            (
                prompts_and_targets_dict,
                text_dataloader,
            ) = src.utils.create_text_dataloader(
                vlm_ensemble=vlm_ensemble,
                prompt_and_targets_kwargs=wandb_config["prompt_and_targets_kwargs"],
                wandb_config=wandb_config,
                split="eval",
                load_prompts_and_targets_kwargs={
                    "train_indices": run_jailbreak_dict["wandb_run_train_indices"],
                },
            )

            # Read image from disk. This image data should match the uint8 images.
            # Shape: Batch-Channel-Height-Width
            adv_image = (
                torchvision.transforms.v2.functional.pil_to_tensor(
                    Image.open(run_jailbreak_dict["file_path"], mode="r")
                ).unsqueeze(0)
                / 255.0
            )

            # Measure and log metrics of model on attacks.
            model_evaluation_results = attacker.evaluate_jailbreak_against_vlms_and_log(
                vlm_ensemble=vlm_ensemble,
                image=adv_image,
                prompts_and_targets_dict=prompts_and_targets_dict,
                text_dataloader=text_dataloader,
                # harmbench_evaluator=None,  # harmbench_evaluator,
                # llamaguard_evalutor=None,  # llamaguard_evalutor,
                wandb_logging_step_idx=run_jailbreak_dict["n_gradient_steps"],
            )[model_name_str]

            model_evaluation_results["eval_model_str"] = model_name_str
            model_evaluation_results["wandb_run_id"] = run_jailbreak_dict[
                "wandb_run_id"
            ]
            model_evaluation_results["wandb_logging_step"] = run_jailbreak_dict[
                "wandb_logging_step"
            ]
            model_evaluation_results["n_gradient_steps"] = run_jailbreak_dict[
                "n_gradient_steps"
            ]
            model_evaluation_results["attack_models_str"] = run_jailbreak_dict[
                "attack_models_str"
            ]

            # Log the evaluation results.
            wandb.log(model_evaluation_results)

            print(
                f"Evaluated {model_name_str} on {run_jailbreak_dict['wandb_run_id']} at step {run_jailbreak_dict['n_gradient_steps']}"
            )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    evaluate_vlm_adversarial_examples()
