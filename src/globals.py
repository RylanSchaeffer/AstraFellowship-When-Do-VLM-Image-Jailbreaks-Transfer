default_config = {
    "attack_kwargs": {
        "attack_name": "ssa_common_weakness",
        "epsilon": 16.0 / 255.0,
        "step_size": 1.0 / 255.0,
        "total_steps": 50,
    },
    "image_initialization": "NIPS17",
    # "models_to_attack": "['blip2-opt-2.7b']",
    "models_to_attack": "['instruct_blip2-opt-2.7b']",
    "models_to_eval": "['instruct_blip2-opt-2.7b']",
    "prompts_and_targets": "advbench",
    "n_samples": 17,
    "seed": 0,
}
