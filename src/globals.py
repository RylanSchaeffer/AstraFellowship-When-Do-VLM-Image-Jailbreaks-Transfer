default_config = {
    "attack_kwargs": {
        "attack_name": "sgd",
        # "attack_name": "ssa_common_weakness",
        "step_size": 0.01,
        "total_steps": 5000,
    },
    "image_initialization": "NIPS17",
    "models_to_attack": "['blip2-opt-2.7b']",
    # "models_to_attack": "{'instructblip2-vicuna-7b'}",
    "models_to_eval": "{'blip2-opt-2.7b'}",
    # "models_to_eval": "{'instructblip2-vicuna-7b'}",
    "prompts_and_targets": "advbench",
    "n_prompts_and_targets": 53,
    "n_samples": 15,
    "seed": 0,
}
