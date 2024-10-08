default_attack_config = {
    # "compile": True,
    "compile": False,
    "data": {
        # "dataset": "all_model_generated_evals",
        "dataset": "wealth",
        "batch_size": 2,
        "num_workers": 1,
        "prefetch_factor": 4,
        "split": "train",
    },
    "image_kwargs": {
        "image_size": 512,
        "image_initialization": "random",
        # "image_initialization": "trina",
    },
    "lightning_kwargs": {
        "accumulate_grad_batches": 6,
        "gradient_clip_val": 10.0,
        # "limit_train_batches": 1.0,
        # "limit_train_batches": 0.05,  # Fast debugging.
        "log_loss_every_n_steps": 1,
        "log_image_every_n_steps": 100,
        "precision": "bf16-mixed",
        # "precision": "bf16-true",
    },
    # "models_to_attack": "{'llava-v1p5-vicuna7b'}",
    # "models_to_attack": "{'llava-v1p6-mistral7b'}",
    # "models_to_attack": "{'llava-v1.6-vicuna13b'}",
    # "models_to_attack": "{'idefics2-8b'}",
    # "models_to_attack": "{'deepseek-vl-1.3b-chat'}",
    # "models_to_attack": "{'deepseek-vl-7b-chat'}",
    # "models_to_attack": "{'xgen-mm-phi3-mini-instruct-r-v1'}",
    "models_to_attack": "{'Qwen-VL-Chat', 'prism-reproduction-llava-v15+7b', 'deepseek-vl-7b-chat', 'prism-clip+7b', 'prism-dinosiglip+7b', 'prism-siglip+7b', 'xgen-mm-phi3-mini-instruct-r-v1'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+13b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'prism-clip+7b'}",
    # "models_to_attack": "{'prism-clip+7b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-reproduction-llava-v15+13b'}",
    # "models_to_attack": "{'prism-clip+7b', 'prism-siglip+7b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'llava-v1p5-vicuna7b', 'llava-v1p6-mistral7b'}",
    "model_generation_kwargs": {},
    "n_grad_steps": 30000,
    "n_generations": 2,
    "optimization": {
        "eps": 1e-4,
        "learning_rate": 0.005,
        # "momentum": 0.0,
        "momentum": 0.9,
        "optimizer": "adam",
        # "optimizer": "sgd",
        "weight_decay": 0.00001,
    },
    "seed": 0,
}


default_eval_config = {
    "data": {
        "batch_size": 1,
        "dataset": "wealth",
        # "num_workers": 1,
        # "num_workers": 2,
        "num_workers": 4,
        # "prefetch_factor": 1,
        # "prefetch_factor": 4,
        "prefetch_factor": 8,
        "split": "eval",
    },
    "lightning_kwargs": {
        # "limit_eval_batches": 1.0,
        # "limit_eval_batches": 0.1,  # Fast debugging.
        "log_loss_every_n_steps": 1,
        "precision": "bf16-mixed",
    },
    # "model_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    "model_to_eval": "{'xgen-mm-phi3-mini-instruct-r-v1'}",
    "model_generation_kwargs": {
        "xgen-mm-phi3-mini-instruct-r-v1": {
            "temperature": 0.0,
            "top_p": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
        },
        "Qwen-VL-Chat": {
            "temperature": 0.0,
            "top_p": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
        },
        "deepseek-vl-1.3b-chat": {
            "temperature": 0.0,
            "top_p": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
        },
        "deepseek-vl-7b-chat": {
            "temperature": 0.0,
            "top_p": 0.0,
            "max_new_tokens": 1,
            "min_new_tokens": 1,
        },
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
    "n_generations": 200,
    "seed": 0,
    "wandb_attack_run_id": "g0jubjzl",
    # "wandb_sweep_id": "yvqszl4d",
}
