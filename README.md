# When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?

This repository contains code and figures for our paper
[When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?](https://www.arxiv.org/abs/2407.15211).

[![arXiv](https://img.shields.io/badge/arXiv-2407.15211-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2407.15211)


[**Installation**](#installation) | [**Usage**](#usage) | [**Training New VLMs**](#training-new-vlms) | [**Citation**](#citation) | [**Contact**](#contact)


## Installation

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

8. Install Prismatic and (optionally) Deepseek (currently broken):

Adding `--config-settings editable_mode=compat` is optionable - its for your vscode language to recognize the packages
`cd submodules/prismatic-vlms && pip install -e . --config-settings editable_mode=compat && cd ../..`
`cd submodules/DeepSeek-VL && pip install -e . --config-settings editable_mode=compat && cd ../..`

9. Then follow their instructions:

`pip install packaging ninja && pip install flash-attn --no-build-isolation`

10. Manually install a few additional packages:

`conda install joblib pandas matplotlib seaborn black tiktoken nvidia-htop sentencepiece hf_transfer anthropic termcolor -y`

11. Make sure to log in to W&B by running `wandb login`
12. Login to Huggingface with `huggingface-cli login`


## Usage

There are 3-4 main components to this repository:

1. Optimizing image jailbreaks against sets of VLMs: [optimize_jailbreak_attacks_against_vlms.py](optimize_jailbreak_attacks_against_vlms.py).
2. Evaluating the transfer of jailbreaks to new VLMs: [evaluate_jailbreak_attacks_against_vlms.py](evaluate_jailbreak_attacks_against_vlms.py).
3. Setting/sweeping hyperparameters for both. This is done in two places. By default, the hyperparameters are set in [globals.py](src/globals.py) but can be overwritten with [W&B sweeps](sweeps). 
4. Evaluating the results in [notebooks](notebooks).

The project is built primarily on top of [PyTorch](https://pytorch.org/), [Lightning](https://lightning.ai/docs/pytorch/stable/), [W&B](https://wandb.ai) and the [Prismatic suite of VLMs](https://github.com/TRI-ML/prismatic-vlms).

## Training New VLMs

Our work was based on the [Prismatic suite of VLMs](https://github.com/TRI-ML/prismatic-vlms)
by Siddharth Karamcheti and collaborators.
To train additional VLMs based on new language models (e.g., Llama 3), we created a [Prismatic fork](https://github.com/RylanSchaeffer/prismatic-vlms).
The new VLMs are [publicly available on HuggingFace](https://huggingface.co/RylanSchaeffer/prismatic-vlms)
and include the following vision backbones:

- CLIP
- SigLIP
- DINOv2

and the following language models:

- Gemma Instruct 2B
- Gemma Instruct 8B
- Llama 2 Chat 7B
- Llama 3 Instruct 8B
- Mistral Instruct v0.2 7B
- Phi 3 Instruct 4B

## Citation

To cite this work, please use:

```bibtex
@article{schaeffer2024universaltransferableimagejailbreaks,
  title={When Do Universal Image Jailbreaks Transfer Between Vision-Language Models?},
  author={Schaeffer, Rylan and Valentine, Dan and Bailey, Luke and Chua, James and Eyzaguirre, Crist{\'o}bal and Durante, Zane and Benton, Joe and Miranda, Brando and Sleight, Henry and Hughes, John and others},
  journal={arXiv preprint arXiv:2407.15211},
  year={2024}
}
```

## Contact

Questions? Comments? Interested in collaborating?
Open an issue or email rschaef@cs.stanford.edu or any of the other authors.