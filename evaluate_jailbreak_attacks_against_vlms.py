from accelerate import Accelerator
import ast
import json
import os
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
    assert len(cuda_visible_devices) == 3

    # Convert these strings to sets of strings.
    # This needs to be done after writing JSON to disk because sets are not JSON serializable.
    wandb_config["models_to_eval"] = ast.literal_eval(wandb_config["models_to_eval"])

    src.utils.set_seed(seed=wandb_config["seed"])

    api = wandb.Api()
    sweep = api.sweep("universal-vlm-jailbreak/" + wandb_config["wandb_sweep_id"])
    runs = list(sweep.runs)
    jailbreak_image_file_paths = []
    for run in runs:
        for file in run.files():
            file_name = str(file.name)
            if not file_name.endswith(".png"):
                continue
            file_dir_path = os.path.join(
                "eval_data", f"sweep={wandb_config['wandb_sweep_id']}", run.id
            )
            os.makedirs(file_dir_path, exist_ok=True)
            file.download(root=file_dir_path, replace=True)
            file_path = os.path.join(file_dir_path, file_name)
            jailbreak_image_file_paths.append(file_path)

    harmbench_evaluator = HarmBenchEvaluator(
        device=int(cuda_visible_devices[-1]),
    )
    llamaguard_evalutor = LlamaGuardEvaluator(
        device=int(cuda_visible_devices[-2]),
    )

    accelerator = Accelerator()

    for model_str in wandb_config["models_to_eval"]:
        vlm_ensemble: VLMEnsemble = src.utils.instantiate_vlm_ensemble(
            model_strs=[model_str],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            accelerator=accelerator,
        )

        # We need to load the VLMs ensemble in order to tokenize the dataset.
        prompts_and_targets_dict, text_dataloader = src.utils.create_text_dataloader(
            vlm_ensemble=vlm_ensemble,
            prompt_and_targets_kwargs=wandb_config["prompt_and_targets_kwargs"],
            wandb_config=wandb_config,
            split="test",
        )

        if wandb_config["compile"]:
            vlm_ensemble: VLMEnsemble = torch.compile(
                vlm_ensemble,
                mode="default",  # good balance between performance and overhead
                # mode="reduce-overhead",  # not guaranteed to work, but good for small batches.
            )

        assert wandb_config["attack_kwargs"]["attack_name"] == "eval"
        attacker = src.utils.create_attacker(
            wandb_config=wandb_config,
            vlm_ensemble=vlm_ensemble,
            accelerator=accelerator,
        )

        for jailbreak_image_file_path in jailbreak_image_file_paths:
            # Read image from disk.
            # TODO: This is probably incorrect.
            adv_image = torch.load(jailbreak_image_file_path)

            # Measure and log metrics of model on attacks.
            attacker.evaluate_jailbreak_against_vlms_and_log(
                vlm_ensemble=vlm_ensemble,
                adv_image=adv_image,
                prompts_and_targets_dict=prompts_and_targets_dict,
                text_dataloader=text_dataloader,
                harmbench_evaluator=harmbench_evaluator,
                llamaguard_evalutor=llamaguard_evalutor,
            )


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    evaluate_vlm_adversarial_examples()
