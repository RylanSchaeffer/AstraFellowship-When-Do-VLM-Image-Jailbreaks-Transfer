# Rylan's READMe

## Notes

- Llava disables gradients for the "vision tower"; see https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/issues/9#issuecomment-1962315340 for the solution
  - Commenting off `llava/models/multimodal_encoder/clip_encoder/line39` should work
- 
