default_config = {
    "attack_kwargs": {
        "attack_name": "ssa_common_weakness",
        "epsilon": 16.0 / 255.0,
        "step_size": 1.0 / 255.0,
        "total_steps": 5,
    },
    "image_initialization": "NIPS17",
    "models_to_attack": "['instruct_blip','blip2']",
    "models_to_test": "['minigpt4']",
    "prompt_text": "robust_bard",
    "target_text": "robust_bard",
    "seed": 0,
}
