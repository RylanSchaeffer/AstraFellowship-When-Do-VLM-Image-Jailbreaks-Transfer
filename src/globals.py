default_config = {
    "attack_kwargs": {
        "attack_name": "sgd",
        # "attack_name": "ssa_common_weakness",
        "step_size": 0.001,
        "total_steps": 1000,
    },
    "image_initialization": "NIPS17",
    "models_to_attack": "['blip2-opt-2.7b']",
    # "models_to_attack": "{'instructblip2-vicuna-7b'}",
    "models_to_eval": "{'blip2-opt-2.7b'}",
    # "models_to_eval": "{'instructblip2-vicuna-7b'}",
    "prompts_and_targets": "advbench",
    "n_samples": 53,
    "seed": 0,
}
