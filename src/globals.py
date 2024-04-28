default_attack_config = {
    # "compile": True,
    "compile": False,
    "data": {
        "dataset": "mmlu",
        "batch_size": 1,
        "num_workers": 1,
        "split": "train",
    },
    "image_kwargs": {
        "image_size": 512,
        # "image_initialization": "random",
        "image_initialization": "trina",
    },
    "lightning_kwargs": {
        "accumulate_grad_batches": 6,
        "gradient_clip_val": 10.0,
        "limit_train_batches": 1.0,
        # "limit_train_batches": 0.05,  # Fast debugging.
        "log_loss_every_n_steps": 1,
        "log_image_every_n_steps": 1000,
        "precision": "bf16-mixed",
        # "precision": "bf16-true",
    },
    # "models_to_attack": "{'llava-v1p5-vicuna7b'}",
    # "models_to_attack": "{'llava-v1p6-mistral7b'}",
    # "models_to_attack": "{'llava-v1.6-vicuna13b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b'}",
    "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'prism-clip+7b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-clip+7b'}",
    # "models_to_attack": "{'prism-clip+7b', 'prism-siglip+7b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'llava-v1p5-vicuna7b', 'llava-v1p6-mistral7b'}",
    "model_generation_kwargs": {},
    "n_grad_steps": 10_000,
    "prompts_and_targets_kwargs": {
        # "dataset_train": "rylan_anthropic_hhh",
        # "dataset_train": "truthfulqa",
        # "n_unique_prompts_and_targets": -1,  # -1 means use all prompts and targets.
        "n_unique_prompts_and_targets": 1_000,  # -1 means use all prompts and targets.
    },
    "optimization": {
        "eps": 1e-4,
        # "learning_rate": 0.001,
        "learning_rate": 0.02,
        "momentum": 0.9,
        "optimizer": "adam",
        "weight_decay": 0.00001,
    },
    "seed": 0,
}


default_eval_config = {
    "data": {
        "num_workers": 1,
        "batch_size": 1,
        "dataset": "mmlu",
        "split": "eval",
    },
    "lightning_kwargs": {
        "log_loss_every_n_steps": 1,
        "precision": "bf16-mixed",
    },
    "n_generations": 100,
    # "model_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    "model_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    # "model_to_eval": "{'prism-clip+7b'}",
    # "model_to_eval": "{'prism-reproduction-llava-v15+7b', 'prism-reproduction-llava-v15+13b'}",
    "model_generation_kwargs": {
        "prism-reproduction-llava-v15+7b": {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
        },
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
    "prompts_and_targets_kwargs": {
        "dataset_eval": "james_truthfulqa",
    },
    "seed": 0,
    "wandb_run_id": "lywump1t",
    # "wandb_sweep_id": "yvqszl4d",
}
