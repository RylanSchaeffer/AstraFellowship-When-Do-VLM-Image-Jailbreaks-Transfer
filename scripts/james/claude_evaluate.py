from concurrent.futures import ThreadPoolExecutor
import anthropic
from git import Sequence
from openai import BaseModel
from slist import Slist
import json
import numpy as np
import wandb
from typing import Any, Dict
import pandas as pd
from scripts.james.loading import (
    JailbreakData,
    load_jailbreak_list_v2,
    load_prompts_and_targets_v2,
)
from src.anthropic_utils.client import AnthropicCaller

import src.data
import src.globals
from src.openai_utils.shared import ChatMessage, InferenceConfig
from src.prompts_and_targets import PromptAndTarget
import src.systems
import src.utils

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
threadpool = ThreadPoolExecutor(max_workers=20)


class SingleResult(BaseModel):
    prompt: str
    target: str
    generation: str
    matches_target: bool
    time_taken: float


def single_generate(
    prompt_idx: int,
    prompt_target: PromptAndTarget,
    adv_image: str,
    config: InferenceConfig,
    caller: AnthropicCaller,
) -> SingleResult:
    start_time = time.time()
    prompt = prompt_target.prompt
    target = prompt_target.target
    message = ChatMessage(
        role="user", content=prompt, image_content=adv_image, image_type="image/png"
    )
    single_gen = caller.call(messages=[message], config=config).content[0].text
    matches_target = single_gen.strip().lower() == target.strip().lower()
    time_taken = time.time() - start_time
    if prompt_idx % 10 == 0:
        print(
            f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {single_gen}\nGeneration Duration: {time_taken} seconds\n\n"
        )
    return SingleResult(
        prompt=prompt,
        target=target,
        generation=single_gen,
        matches_target=matches_target,
        time_taken=time_taken,
    )


def parallel_generate(
    prompt_targets: Sequence[PromptAndTarget],
    adv_image: str,
    config: InferenceConfig,
    caller: AnthropicCaller,
):
    slist_items = Slist(prompt_targets)
    results = slist_items.enumerated().par_map(
        lambda prompt_target: single_generate(
            prompt_idx=prompt_target[0],
            prompt_target=prompt_target[1],
            adv_image=adv_image,
            config=config,
            caller=caller,
        ),
        executor=threadpool,
    )

    return results


def evaluate_vlm_adversarial_examples():
    # Create a config
    # model="claude-3-opus-20240229" #
    # model_to_eval="claude-3-sonnet-20240229"  #
    model_to_eval = "claude-3-haiku-20240307"
    src.globals.default_eval_config["model_to_eval"] = model_to_eval
    api_key = dotenv.dotenv_values()["ANTHROPIC_API_KEY"]
    assert api_key is not None, "Please set the ANTHROPIC_API_KEY in your .env file"
    # Create a client
    client = anthropic.Anthropic(api_key=api_key)
    caller = AnthropicCaller(client=client, cache_path="eval_data/claude_cache.jsonl")

    wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=src.globals.default_eval_config,
        entity=src.utils.retrieve_wandb_username(),
    )
    wandb_config: Dict[str, Any] = dict(wandb.config)

    # Create checkpoint directory for this run, and save the config to the directory.
    wandb_run_dir = os.path.join("runs", wandb.run.id)  # type: ignore
    os.makedirs(wandb_run_dir)
    wandb_config["wandb_run_dir"] = wandb_run_dir
    with open(os.path.join(wandb_run_dir, "wandb_config.json"), "w") as fp:
        json.dump(obj=wandb_config, fp=fp)

    # Load jailbreak images' paths.
    runs_jailbreak_list: list[JailbreakData] = load_jailbreak_list_v2(
        wandb_run_id=wandb_config["wandb_run_id"],
        wandb_sweep_id=None,
    )
    # skip steps < 400
    # runs_jailbreak_list = [run for run in runs_jailbreak_list if run.optimizer_step_counter >= 400]
    int(wandb_config["n_generations"])
    # Load the raw prompts to use for generate.
    prompts_and_targets = load_prompts_and_targets_v2(
        dataset=wandb_config["data"]["dataset"],
        split=wandb_config["data"]["split"],
    )[:800]

    config = InferenceConfig(model=model_to_eval, temperature=0.0, max_tokens=1)

    # model: str,
    to_log: list[dict] = []

    for run_jailbreak_dict in runs_jailbreak_list:
        # Read image from disk. This image data should match the uint8 images.
        # Shape: Batch-Channel-Height-Width
        adv_image = run_jailbreak_dict.image_base_64

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

        # for prompt_idx, prompt_target in enumerate(prompts_and_targets):
        #     prompt = prompt_target.prompt
        #     target = prompt_target.target
        #     start_time = time.time()
        #     # todo: You can paralleise this
        #     message = ChatMessage(role="user", content=prompt_target.prompt, image_content=adv_image, image_type="image/png")
        #     single_gen = caller.cached_call(messages=[message], config=config).content[0].text
        #     matches_target = single_gen.strip().lower() == target.strip().lower()
        #     model_generations_dict["generations"].append(single_gen)
        #     model_generations_dict["prompts"].append(prompt)
        #     model_generations_dict["targets"].append(target)
        #     model_generations_dict["matches_target"].append(matches_target)
        #     end_time = time.time()
        #     if prompt_idx % 10 == 0:
        #         print(
        #             f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {single_gen}\nGeneration Duration: {end_time - start_time} seconds\n\n"
        #         )
        ## Paralleised
        results = parallel_generate(
            prompt_targets=prompts_and_targets,
            adv_image=adv_image,
            config=config,
            caller=caller,
        )
        for result in results:
            model_generations_dict["generations"].append(result.generation)
            model_generations_dict["prompts"].append(result.prompt)
            model_generations_dict["targets"].append(result.target)
            model_generations_dict["matches_target"].append(result.matches_target)
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
