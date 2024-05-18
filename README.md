# Perez Astra Fellowship Universal & Transferable VLM Jailbreaks

## Setup

1. (Optional) Update conda:

`conda update -n base -c defaults conda -y`

2. Create and activate the conda environment:

`conda create -n universal_vlm_jailbreak_env python=3.11 -y && conda activate universal_vlm_jailbreak_env`

4. Update pip in preparation of the next step.

`pip install --upgrade pip`

5. Install Pytorch:

`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`

6. Install Lightning:

`conda install lightning -c conda-forge -y`

7. Grab the git submodules:

`git submodule update --init --recursive`

8. Install Prismatic and Deepseek

Adding `--config-settings editable_mode=compat` is optionable - its for your vscode language to recognize the packages
`cd submodules/prismatic-vlms && pip install -e . --config-settings editable_mode=compat && cd ../..`
`cd submodules/DeepSeek-VL && pip install -e . --config-settings editable_mode=compat && cd ../..`


9. Then follow their instructions:

`pip install packaging ninja && pip install flash-attn --no-build-isolation`

10. Manually install a few additional packages:

`conda install joblib pandas matplotlib seaborn black tiktoken -y`

10a. For running salesforce's xgen
```
# Needs the latest dev version("4.41.0.dev0") 
pip uninstall -y transformers && pip install git+https://github.com/huggingface/pip
pip install open_clip_torch==2.24.0
pip install einops
pip install einops-exts
```

11. Install more stuff. 

`pip install nvidia-htop sentencepiece hf_transfer`

12. Make sure to log in to W&B by running `wandb login`
13. Login to Huggingface with `huggingface-cli login`


Note: To run on a CPU-only machine (e.g., for eval), use `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

Note: You might need to subsequently run `conda install conda-forge::charset-normalizer -y`

### Additional Modifications

- I am using prismatic `41a93d1295ca18e593223f6cde50506db7728ec7`
- Prismatic VLMs also disables gradients for the vision backbone. Comment out https://github.com/TRI-ML/prismatic-vlms/blob/main/prismatic/models/vlms/prismatic.py#L308 to optimize attacks.
- Prismatic VLMs also assumes a default `HF_HUB_REPO=TRI-ML/prismatic-vlms`. Until Sidd adds our models to his HF repo, we need to add a conditional to `prismatic/models/load.py` to handle our models. Go to line 58: 

```python
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )
```

Replace the above python code with the below python code:

```python
        rylan_trained_models = {
            "prism-gemma-instruct+2b+clip",
            "prism-gemma-instruct+2b+siglip",
            "prism-gemma-instruct+2b+dinosiglip",
            "prism-gemma-instruct+8b+clip",
            "prism-gemma-instruct+8b+siglip",
            "prism-gemma-instruct+8b+dinosiglip",
            "prism-llama2-chat+7b+clip",
            "prism-llama2-chat+7b+siglip",
            "prism-llama2-chat+7b+dinosiglip",
            "prism-llama3-instruct+8b+clip",
            "prism-llama3-instruct+8b+siglip",
            "prism-llama3-instruct+8b+dinosiglip",
            "prism-mistral-instruct-v0.2+7b+clip",
            "prism-mistral-instruct-v0.2+7b+siglip",
            "prism-mistral-instruct-v0.2+7b+dinosiglip",
            "prism-phi-instruct-3+4b+clip",
            "prism-phi-instruct-3+4b+siglip",
            "prism-phi-instruct-3+4b+dinosiglip",
        }
        
        if (model_id_or_path not in rylan_trained_models) and (model_id_or_path not in GLOBAL_REGISTRY):
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        if model_id_or_path in rylan_trained_models:
            HF_HUB_REPO = "RylanSchaeffer/prismatic-vlms"
        else:
            HF_HUB_REPO = "TRI-ML/prismatic-vlms"

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )
```

- `timm` must be version `0.9.16` not `1.0.3`
- Llava disables gradients for the "vision tower"; see https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/issues/9#issuecomment-1962315340 for the solution
  - Commenting off `llava/models/multimodal_encoder/clip_encoder/line39` should work


## Optimizing Jailbreaks

Create a sweep using `wandb sweep <path to W&B sweep e.g., sweeps/attack/...>`. This will give you a W&B sweep ID. Then:

```
cd PerezAstraFellowship-Universal-VLM-Jailbreak
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=7
wandb agent rylan/universal-vlm-jailbreak/srvmywo4
```


## Evaluating Jailbreaks

```
cd PerezAstraFellowship-Universal-VLM-Jailbreak
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
wandb agent rylan/universal-vlm-jailbreak-eval/jb02fx4o
```


## Runpod IO Setup


```bash
apt-get update && apt-get install vim -y && apt-get install nano -y && apt-get install tmux
cd /workspace && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && chmod +x Miniconda3-latest-Linux-x86_64.sh && ./Miniconda3-latest-Linux-x86_64.sh
```

Create SSH key and register it with GitHub:
```bash
ssh-keygen -t ed25519 -C "rylanschaeffer@gmail.com"
mkdir /workspace/.ssh && cp /root/.ssh/* /workspace/.ssh/ && ls /workspace/.ssh/
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


1. `pip install anthropic termcolor`
2. Make `SECRETS`"
3. `export LFS_HOME=/workspace`