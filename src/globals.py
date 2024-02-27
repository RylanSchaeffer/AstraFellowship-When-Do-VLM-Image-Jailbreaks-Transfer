default_config = {
    "attack_kwargs": {
        "attack_name": "pgd",
        # "attack_name": "ssa_common_weakness",
        # "clip_gradients": True,
        # "normalize_gradients": True,
        "step_size": 0.01,
        "batch_size": 32,
        "total_steps": 5000,
        "generate_every_n_steps": 100,
    },
    "image_initialization": "random",
    # "models_to_attack": "{'blip2-opt-2.7b'}",
    "models_to_attack": "{'llava-v1.5-7b'}",
    "models_to_eval": "{'llava-v1.5-7b'}",
    "model_generation_kwargs": {
        "llava-v1.5-7b": {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_new_tokens": 1024,
        },
    },
    # "models_to_eval": "{'instructblip2-vicuna-7b'}",
    "prompts_and_targets": "advbench",
    "n_prompts_and_targets": 53,
    "datum_index": 0,
    "seed": 0,
}
