default_attack_config = {
    "attack_kwargs": {
        "attack_name": "pgd",
        # "attack_name": "ssa_common_weakness",
        "batch_size": 2,
        # "clip_gradients": True,
        # "normalize_gradients": True,
        "log_image_every_n_steps": 5,
        "step_size": 0.01,
        "total_steps": 10,
    },
    "compile": True,
    "data": {
        "num_workers": 4,
        "shuffle_train": True,
        "shuffle_test": False,
    },
    "image_kwargs": {
        "image_size": 512,
        "image_initialization": "random",
    },
    # "models_to_attack": "{'llava-v1p5-vicuna7b'}",
    # "models_to_attack": "{'llava-v1p6-mistral7b'}",
    # "models_to_attack": "{'llava-v1.6-vicuna13b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b'}",
    "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-clip+7b'}",
    # "models_to_attack": "{'prism-clip+7b', 'prism-siglip+7b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'llava-v1p5-vicuna7b', 'llava-v1p6-mistral7b'}",
    # TODO: Move these model generation kwargs into the model classes to be overwritten.
    "model_generation_kwargs": {
        "llava-v1p6-mistral7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p5-vicuna7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p6-vicuna7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p6-vicuna13b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
            "use_cache": False,
            "cache_position": None,
        },
        "prism-reproduction-llava-v15+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-reproduction-llava-v15+13b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-clip+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-siglip+7b": {
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
        "llava-lvis4v-lrv+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
    },
    "prompt_and_targets_kwargs": {
        "dataset_train": "rylan_anthropic_hhh",
        # "dataset_train": "advbench",
        "dataset_test": "advbench",
        "n_unique_prompts_and_targets": -1,  # -1 means use all prompts and targets.
    },
    "seed": 0,
}


default_eval_config = {
    "attack_kwargs": {
        "attack_name": "eval",
    },
    "models_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    "model_generation_kwargs": {
        "llava-v1p6-mistral7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p5-vicuna7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p6-vicuna7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "llava-v1p6-vicuna13b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
            "use_cache": False,
            "cache_position": None,
        },
        "prism-reproduction-llava-v15+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-reproduction-llava-v15+13b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-clip+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
        "prism-siglip+7b": {
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
        "llava-lvis4v-lrv+7b": {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 100,
            "min_new_tokens": 5,
        },
    },
    "seed": 0,
    "wandb_sweep_id": "3332ohsb",
}
