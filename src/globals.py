from collections import OrderedDict


DATASETS_TO_NICE_STRINGS_DICT = {
    "advbench": "AdvBench",
    "rylan_anthropic_hhh": "Anthropic HHH",
}

METRICS_TO_TITLE_STRINGS_DICT = {
    "loss/avg_epoch": "Cross Entropy",
    "loss/score_model=claude3opus": "Claude 3 Opus",
    "loss/score_model=harmbench": "HarmBench",
    "loss/score_model=llamaguard2": "LlamaGuard2",
    "one_minus_score_model=claude3opus": "Claude 3 Opus",
    "one_minus_score_model=harmbench": "HarmBench",
    "one_minus_score_model=llamaguard2": "LlamaGuard2",
}

METRICS_TO_LABELS_NICE_STRINGS_DICT = {
    "loss/avg_epoch": "Loss",
    "loss/score_model=claude3opus": "Attack Success Rate",
    "loss/score_model=harmbench": "Attack Success Rate",
    "loss/score_model=llamaguard2": "Attack Success Rate",
    "one_minus_score_model=claude3opus": "Attack Failure Rate",
    "one_minus_score_model=harmbench": "Attack Failure Rate",
    "one_minus_score_model=llamaguard2": "Attack Failure Rate",
}


METRICS_TO_BOUNDS_DICT = {
    "loss/avg_epoch": (0.0, None),
    "loss/score_model=claude3opus": (0.0, 1.0),
    "loss/score_model=harmbench": (0.0, 1.0),
    "loss/score_model=llamaguard2": (0.0, 1.0),
    "one_minus_score_model=claude3opus": (0.0, 1.0),
    "one_minus_score_model=harmbench": (0.0, 1.0),
    "one_minus_score_model=llamaguard2": (0.0, 1.0),
}

MODELS_TO_NICE_STRINGS_DICT = {
    "prism-reproduction-llava-v15+7b": "LLAVAv1.5+7B (Reproduction)",
    "prism-reproduction-llava-v15+13b": "LLAVAv1.5+13B (Reproduction)",
}


default_attack_config = {
    # "compile": True,
    "compile": False,
    "data": {
        "dataset": "advbench",
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 4,
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
        # "limit_train_batches": 1.0,
        "limit_train_batches": 0.05,  # Fast debugging.
        "log_loss_every_n_steps": 1,
        "log_image_every_n_steps": 1000,
        "precision": "bf16-mixed",
        # "precision": "bf16-true",
    },
    # "models_to_attack": "{'llava-v1p5-vicuna7b'}",
    # "models_to_attack": "{'llava-v1p6-mistral7b'}",
    # "models_to_attack": "{'llava-v1.6-vicuna13b'}",
    # "models_to_attack": "{'idefics2-8b'}",
    # "models_to_attack": "{'deepseek-vl-1.3b-chat'}",
    # "models_to_attack": "{'deepseek-vl-7b-chat'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b'}",
    # "models_to_attack": "{'Qwen-VL-Chat'}",
    # "models_to_attack": "{'xgen-mm-phi3-mini-instruct-r-v1'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+13b'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'prism-clip+7b'}",
    # "models_to_attack": "{'prism-clip+7b'}",
    "models_to_attack": "{'prism-phi-instruct-3+4b+clip'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-clip+7b'}",
    # "models_to_attack": "{'prism-reproduction-llava-v15+7b', 'prism-reproduction-llava-v15+13b'}",
    # "models_to_attack": "{'prism-clip+7b', 'prism-siglip+7b'}",
    # "models_to_attack": "{'prism-gemma-instruct+2b+dinosiglip'}",
    # "models_to_attack": "{'prism-llama3-instruct+8b+dinosiglip'}",
    # "models_to_attack": "{'prism-gemma-instruct+2b+siglip', 'prism-llama3-instruct+8b+dinosiglip'}",
    # "models_to_attack": "{'prism-dinosiglip+7b'}",
    # "models_to_attack": "{'llava-v1p5-vicuna7b', 'llava-v1p6-mistral7b'}",
    "model_generation_kwargs": {},
    "n_grad_steps": 29,
    "n_generations": 2,
    "optimization": {
        "eps": 1e-4,
        "learning_rate": 0.001,
        "momentum": 0.0,
        # "momentum": 0.9,
        # "optimizer": "adam",
        "optimizer": "sgd",
        "weight_decay": 0.00001,
    },
    "seed": 0,
}


default_eval_config = {
    "data": {
        "batch_size": 1,
        "dataset": "rylan_anthropic_hhh",
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
        "limit_eval_batches": 0.1,  # Fast debugging.
        "log_loss_every_n_steps": 1,
        "precision": "bf16-mixed",
    },
    # "model_to_eval": "{'prism-reproduction-llava-v15+7b'}",
    "model_to_eval": "{'prism-clip+7b'}",
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
    "n_generations": 5,
    "seed": 0,
    "wandb_attack_run_id": "llu4vrns",
    # "wandb_sweep_id": "yvqszl4d",
}
