default_attack_config = {
    "attack_kwargs": {
        "attack_name": "pgd",
        # "attack_name": "ssa_common_weakness",
        "batch_size": 2,
        # "clip_gradients": True,
        # "normalize_gradients": True,
        "log_image_every_n_steps": 1000,
        "log_loss_every_n_steps": 25,
        "precision": "bfloat16",
        "step_size": 0.001,
        "total_steps": 10000,
    },
    "compile": True,
    "data": {
        "num_workers": 1,
    },
    "image_kwargs": {
        "image_size": 512,
        "image_initialization": "random",
    },
    # "models_to_attack": "{'llava-v1p5-vicuna7b'}",
    # "models_to_attack": "{'llava-v1p6-mistral7b'}",
    # "models_to_attack": "{'llava-v1.6-vicuna13b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b'}",
    "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-clip+7b'}",
    # "models_to_attack": "{'prism-clip+7b', 'prism-siglip+7b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'llava-v1p5-vicuna7b', 'llava-v1p6-mistral7b'}",
    "model_generation_kwargs": {},
    "prompt_and_targets_kwargs": {
        "dataset_train": "rylan_anthropic_hhh",
        # "n_unique_prompts_and_targets": -1,  # -1 means use all prompts and targets.
        "n_unique_prompts_and_targets": 189,  # -1 means use all prompts and targets.
    },
    "seed": 0,
}


default_eval_config = {
    "attack_kwargs": {
        "attack_name": "eval",
        "batch_size": 1,
    },
    "data": {
        "num_workers": 1,
    },
    "models_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    # "models_to_eval": "{'prism-reproduction-llava-v15+7b', 'prism-reproduction-llava-v15+13b'}",
    "model_generation_kwargs": {
        # "prism-reproduction-llava-v15+7b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
        # "prism-reproduction-llava-v15+13b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
        # "prism-clip+7b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
        # "prism-siglip+7b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
        # "prism-dinosiglip+7b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
        # "llava-lvis4v-lrv+7b": {
        #     "temperature": 0.1,
        #     "top_p": 0.9,
        #     "max_new_tokens": 100,
        #     "min_new_tokens": 5,
        # },
    },
    "prompt_and_targets_kwargs": {
        "dataset_eval": "rylan_anthropic_hhh",
    },
    "seed": 0,
    "wandb_sweep_id": "8090iibg",
}
