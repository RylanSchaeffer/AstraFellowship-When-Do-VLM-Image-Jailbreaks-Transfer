from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from slist import Slist
import json
import numpy as np
import wandb
from typing import Any, Dict, Sequence
import pandas as pd
from scripts.james.loading import (
    JailbreakData,
    load_jailbreak_list_v2,
    load_prompts_and_targets_v2,
)
from src.openai_utils.shared import InferenceConfig

import src.data
import src.globals
from src.openai_utils.shared import ChatMessage
from scripts.james.loading import PromptAndTarget
import src.systems
import src.utils
from src.openai_utils.client import OpenAICachedCaller, OpenaiResponse
from scripts.james.james_globals import default_eval_config
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

# Please set your .env file with the OPENAI_API_KEY
dotenv.load_dotenv()
# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Please provide an OpenAI API Key"
caller = OpenAICachedCaller(
    cache_path="eval_data/gpt_4_cache_v2.jsonl", api_key=api_key
)


class SingleResult(BaseModel):
    prompt: str
    target: str
    target_proba: float
    generation: str
    matches_target: bool
    time_taken: float


# model_to_eval = "gpt-4-turbo"
model_to_eval = "gpt-4o"
config = InferenceConfig(model=model_to_eval, temperature=0.0, max_tokens=1)


def single_generate(
    prompt_idx: int,
    prompt_target: PromptAndTarget,
    adv_image: str,
    caller: OpenAICachedCaller,
) -> SingleResult | None:
    start_time = time.time()
    prompt = prompt_target.prompt
    target = prompt_target.target
    message = ChatMessage(
        role="user", content=prompt, image_content=adv_image, image_type="image/png"
    )
    response: OpenaiResponse = caller.call(messages=[message], config=config)
    single_gen = response.first_response()
    proba_of_target = response.first_token_probability_for_target(target)

    if not single_gen:
        print(f"Failed for {prompt_target}")
        return None
    matches_target = single_gen.strip().lower() == target.strip().lower()
    time_taken = time.time() - start_time
    if prompt_idx % 10 == 0:
        print(
            f"Prompt Idx: {prompt_idx}\nPrompt: {prompt}\nGeneration: {single_gen}\nTarget Proba: {proba_of_target:2f}Generation Duration: {time_taken:2f} seconds\n\n"
        )
    return SingleResult(
        prompt=prompt,
        target=target,
        generation=single_gen,
        matches_target=matches_target,
        time_taken=time_taken,
        target_proba=proba_of_target,
    )


def parallel_generate(
    prompt_targets: Sequence[PromptAndTarget],
    adv_image: str,
    caller: OpenAICachedCaller,
) -> Slist[SingleResult | None]:
    slist_items = Slist(prompt_targets)
    results = slist_items.enumerated().par_map(
        lambda prompt_target: single_generate(
            prompt_idx=prompt_target[0],
            prompt_target=prompt_target[1],
            adv_image=adv_image,
            caller=caller,
        ),
        executor=threadpool,
    )

    return results


def evaluate_vlm_adversarial_examples():
    dotenv.load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    assert openai_key, "Please provide an OpenAI API Key"
    config = default_eval_config
    config["model_to_eval"] = model_to_eval
    wandb.init(
        project="universal-vlm-jailbreak-eval",
        config=config,
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
    runs_jailbreak_list: Slist[JailbreakData] = load_jailbreak_list_v2(
        wandb_run_id=wandb_config["wandb_attack_run_id"],
        wandb_sweep_id=None,
        # Smallest to largest optimizer_step_counter.
    ).sort_by(lambda x: x.optimizer_step_counter)

    int(wandb_config["n_generations"])
    # Load the raw prompts to use for generate.
    prompts_and_targets = load_prompts_and_targets_v2(
        dataset=wandb_config["data"]["dataset"],
        split=wandb_config["data"]["split"],
    )[:600]

    # model: str,
    to_log: list[dict] = []

    for run_jailbreak_dict in runs_jailbreak_list:
        # Read image from disk. This image data should match the uint8 images.
        # Shape: Batch-Channel-Height-Width
        adv_image = run_jailbreak_dict.image_base_64

        wandb_additional_data = {
            "eval_model_str": model_to_eval,
            "wandb_attack_run_id": run_jailbreak_dict.wandb_run_id,
            "optimizer_step_counter": run_jailbreak_dict.optimizer_step_counter,
            "attack_models_str": run_jailbreak_dict.attack_models_str,
        }

        model_generations_dict = {
            "generations": [],
            "prompts": [],
            "targets": [],
            "matches_target": [],
            "target_proba": [],
        }
        results = parallel_generate(
            prompt_targets=prompts_and_targets,
            adv_image=adv_image,
            caller=caller,
        ).flatten_option()
        for result in results:
            model_generations_dict["generations"].append(result.generation)
            model_generations_dict["prompts"].append(result.prompt)
            model_generations_dict["targets"].append(result.target)
            model_generations_dict["matches_target"].append(result.matches_target)
            model_generations_dict["target_proba"].append(result.target_proba)
        model_generations_dict["success_rate"] = np.mean(
            model_generations_dict["matches_target"]
        )
        model_generations_dict["mean_target_proba"] = np.mean(
            model_generations_dict["target_proba"]
        )

        # model_generations_dict["success_rate"] = np.mean(
        #     model_generations_dict["matches_target"]
        # )

        merged_dict = {**wandb_additional_data, **model_generations_dict}

        to_log.append(merged_dict)
        wandb.log(merged_dict)
    # make a pd dataframe
    df = pd.DataFrame(to_log)
    wandb.log({"evaluations": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    evaluate_vlm_adversarial_examples()
