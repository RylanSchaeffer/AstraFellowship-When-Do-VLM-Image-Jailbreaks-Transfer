import torchvision.transforms.v2.functional
import json
import numpy as np
from PIL import Image
import wandb
from typing import Any, Dict
import pandas as pd

import src.data
import src.globals
from src.prompts_and_targets import PromptAndTarget
import src.systems
import src.utils
from src.utils import JailbreakData
from src.openai_utils.client import OpenAIClient
import os
import time
import dotenv


# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str



# async def call_openai_and_log(jailbreak: JailbreakData, prompt: PromptAndTarget, client: OpenAIClient):
#     ...


def evaluate_vlm_adversarial_examples():
    dotenv.load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    assert openai_key, "Please provide an OpenAI API Key"
    client = OpenAIClient(api_key=openai_key)

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

    # Load jailbreak images' paths.
    runs_jailbreak_list: list[JailbreakData] = src.utils.load_jailbreak_list(
        wandb_run_id=wandb_config["wandb_run_id"],
        wandb_sweep_id=None,
        # refresh=True,
        refresh=False,
    )
    n_generations: int = int(wandb_config["n_generations"])
    # Load the raw prompts to use for generate.
    prompts_and_targets = src.data.load_prompts_and_targets_v2(
        dataset=wandb_config["data"]["dataset"],
        split=wandb_config["data"]["split"],
    )[: 100]

    model_to_eval = "gpt-4-turbo"

    # model: str,
    to_log: list[dict] = []
    

    
    for run_jailbreak_dict in runs_jailbreak_list:
        # Read image from disk. This image data should match the uint8 images.
        # Shape: Batch-Channel-Height-Width
        adv_image  = run_jailbreak_dict.image_base_64

        wandb_additional_data = {
            "eval_model_str": model_to_eval,
            "wandb_run_id": run_jailbreak_dict.wandb_run_id,
            "optimizer_step_counter": run_jailbreak_dict.optimizer_step_counter,
            "attack_models_str": run_jailbreak_dict.attack_models_str,
        }

        model_generations_dict = {
            "generations": [],
            "prompts": [],
            "targets": [],
            "matches_target": [],
        }
        
        for prompt_idx, prompt_target in enumerate(prompts_and_targets):
            prompt = prompt_target.prompt
            target = prompt_target.target
            start_time = time.time()
            # todo: You can paralleise this
            single_gen = client.call_gpt_4_turbo(question=prompt_target.prompt, image_base_64=adv_image, max_tokens=1, temperature=0.0)
            matches_target = single_gen.strip().lower() == target.strip().lower()
            model_generations_dict["generations"].append(single_gen)
            model_generations_dict["prompts"].append(prompt)
            model_generations_dict["targets"].append(target)
            model_generations_dict["matches_target"].append(matches_target)
            end_time = time.time()
            if prompt_idx % 10 == 0:
                print(
                    f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {single_gen}\nGeneration Duration: {end_time - start_time} seconds\n\n"
                )
        model_generations_dict["success_rate"] = np.mean(
            model_generations_dict["matches_target"]
        )

        merged_dict = {**wandb_additional_data, **model_generations_dict}

        to_log.append(merged_dict)
        wandb.log(merged_dict)
    # make a pd dataframe
    df = pd.DataFrame(to_log)
    wandb.log({"evaluations": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    evaluate_vlm_adversarial_examples()
