# Perez Astra Fellowship Universal & Transferable VLM Jailbreaks

## Setup

1. (Optional) Update conda:

`conda update -n base -c defaults conda`

2. Create and activate the conda environment:

`conda create -n universal_vlm_jailbreak_env python=3.11 -y && conda activate universal_vlm_jailbreak_env`

4. Update pip in preparation of the next step.

`pip install --upgrade pip`

5. Install Pytorch:

`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

6. Install Lightning:

`conda install lightning -c conda-forge`

7. Grab the git submodules: `git submodule update --init --recursive`

8. Install Prismatic `cd submodules/prismatic-vlms && pip install -e . && cd ../..`.
9. Then follow their instructions:

`pip install packaging ninja`

`pip install flash-attn --no-build-isolation`


10. Manually install a few additional packages:

`conda install joblib pandas matplotlib seaborn nvidia-htop black wandb`

11Make sure to log in to W&B by running `wandb login`.



11. Install more stuff `pip install sentencepiece`

Note: To run on a CPU-only machine (e.g., for eval), use `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

Note: You might need to subsequently run `conda install conda-forge::charset-normalizer`

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
export CUDA_VISIBLE_DEVICES=3
wandb agent rylan/universal-vlm-jailbreak/681o8dt6
```


## Evaluating Jailbreaks

```
cd PerezAstraFellowship-Universal-VLM-Jailbreak
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=6
wandb agent rylan/universal-vlm-jailbreak-eval/poe9qs91
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
We'll clone the repo into the /workspace folder so that runpod doesn't delete it whenever our pod restarts.
```bash
cd /workspace
git clone git@github.com:RylanSchaeffer/PerezAstraFellowship-Universal-VLM-Jailbreak.git`
```


Then modify your `.bashrc` to add:

```
conda activate universal_vlm_jailbreak_env
cd PerezAstraFellowship-Universal-VLM-Jailbreak
```

## Hugginface setup
 these models use llama-2 / llama guard eventually somewhere, 
i had to request  access by filling up meta's form with the same email as your huggingface , according to https://huggingface.co/meta-llama/LlamaGuard-7b/discussions/6
then request for llamas / llama-guard's 
https://huggingface.co/meta-llama/Llama-2-7b-hf. 
https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B

OSError: You are trying to access a gated repo.
Make sure to request access at https://huggingface.co/meta-llama/Llama-2-7b-hf and pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`.

## Do something like symlink
```
Your token has been saved to /root/.cache/huggingface/token
```