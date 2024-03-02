default_config = {
    "attack_kwargs": {
        "attack_name": "pgd",
        # "attack_name": "ssa_common_weakness",
        "batch_size": 1,
        # "clip_gradients": True,
        # "normalize_gradients": True,
        "generate_every_n_steps": 5,
        "step_size": 0.01,
        "total_steps": 10,
    },
    "data_kwargs": {
        "image_size": 336,
    },
    "image_initialization": "random",
    # "models_to_attack": "{'llava-v1.5-vicuna7b'}",
    "models_to_attack": "{'llava-v1.6-mistral7b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_eval": "{'llava-v1.5-vicuna7b'}",
    "models_to_eval": "{'llava-v1.6-mistral7b'}",
    # "models_to_eval": "{'prism-dinosiglip+7b'}",
    "model_generation_kwargs": {
        "llava-v1.6-mistral7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1.5-vicuna7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1.6-vicuna13b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-dinosiglip+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
    },
    # "models_to_eval": "{'instructblip2-vicuna-7b'}",
    "n_unique_prompts_and_targets": -1,  # -1 means use all prompts and targets.
    # "prompts_and_targets": "advbench",
    "prompts_and_targets": "anthropic_hhh",
    "seed": 0,
}
