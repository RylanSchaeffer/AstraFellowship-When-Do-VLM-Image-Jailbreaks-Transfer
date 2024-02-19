default_config = {
    "attack_kwargs": {
        "attack_name": "ssa_common_weakness",
        "epsilon": 16.0 / 255.0,
        "step_size": 1.0 / 255.0,
        "total_steps": 500,
    },
    "image_initialization": "NIPS17",
    "models_to_attack": "['blip2']",
    "models_to_test": "None",
    "prompts_and_targets": "advbench",
    "seed": 0,
}
