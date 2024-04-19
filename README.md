# Perez Astra Fellowship Universal & Transferable VLM Jailbreaks

## Setup

1. (Optional) Update conda:

`conda update -n base -c defaults conda`

2. Create a conda environment:

`conda create -n universal_vlm_jailbreak_env python=3.11`

3. Activate the environment:

`conda activate universal_vlm_jailbreak_env`

4. Update pip in preparation of the next step.

`pip install --upgrade pip`

5. Manually install a few additional packages:

`pip install joblib pandas matplotlib seaborn nvidia-htop black wandb ligthning torchvision`

6. Make sure to log in to W&B by running `wandb login`.

7. Grab the git submodules: `git submodule update --init --recursive`
8. `cd submodules/prismatic-vlms && pip install -e .`.
9. Then follow their instructions:

`pip install packaging ninja`

`pip install flash-attn --no-build-isolation`

11. Install more stuff `pip install sentencepiece`

- Prismatic VLMs also disables gradients for the vision backbone. Disabling https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L308 should work.
- Llava disables gradients for the "vision tower"; see https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/issues/9#issuecomment-1962315340 for the solution
  - Commenting off `llava/models/multimodal_encoder/clip_encoder/line39` should work

### Additional Modifications

- Prismatic VLMs also disables gradients for the vision backbone. Disabling https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L308 is necessary to optimize attacks.
- Llava disables gradients for the "vision tower"; see https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/issues/9#issuecomment-1962315340 for the solution
  - Commenting off `llava/models/multimodal_encoder/clip_encoder/line39` should work


## Optimizing Jailbreaks

Create a sweep using `wandb sweep <path to W&B sweep e.g., sweeps/attack/...>`. This will give you a W&B sweep ID. Then:

```
cd PerezAstraFellowship-Universal-VLM-Jailbreak
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2,3
wandb agent rylan/universal-vlm-jailbreak/<sweep id>
```


## Evaluating Jailbreaks

```
cd PerezAstraFellowship-Universal-VLM-Jailbreak
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
wandb agent rylan/universal-vlm-jailbreak/yvqszl4d
```


## Runpod IO Setup



```bash
apt-get update && apt-get install vim -y && apt-get install nano -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh
```

Create SSH key and register it with GitHub:
```bash
ssh-keygen -t ed25519 -C "rylanschaeffer@gmail.com"
eval "$(ssh-agent -s)"
ssh-add /workspace/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Then go to GitHub and add the SSH key to your account. Once you're setup, clone the repo:

`git clone git@github.com:RylanSchaeffer/PerezAstraFellowship-Universal-VLM-Jailbreak.git`

Then modify your `.bashrc` to add:

```
conda activate universal_vlm_jailbreak_env
cd PerezAstraFellowship-Universal-VLM-Jailbreak
```
