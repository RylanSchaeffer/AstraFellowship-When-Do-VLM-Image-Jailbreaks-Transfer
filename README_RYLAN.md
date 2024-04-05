# Rylan's README

```bash
conda activate universal_vlm_jailbreak_env
cd PerezAstraFellowship-Image-DAN-VLM-Attack/
export CUDA_VISIBLE_DEVICES=4
wandb agent rylan/universal-vlm-jailbreak/idfc9ms6
```

## Notes

- Llava disables gradients for the "vision tower"; see https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/issues/9#issuecomment-1962315340 for the solution
  - Commenting off `llava/models/multimodal_encoder/clip_encoder/line39` should work
- Prismatic VLMs also disables gradients for the vision backbone. Disabling https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L308 should work.

