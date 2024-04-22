# Prompts and Targets

## Data Sourcing

- [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main) `advbench`
  - `harmful_behaviors.csv`: Copied from https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv on 2024/02/19
- 
- [How Robust is Google's Bard to Adversarial Image Attacks?](https://github.com/thu-ml/Attack-Bard/) `robust_bard`


Note! I needed `datasets==2.8.0` in order to download the Anthropic dataset. Otherwise it threw this funky error:

https://huggingface.co/datasets/Anthropic/hh-rlhf/discussions/14