# Attack-Bard

## News

---

[2023/10/14] We have updated the results on GPT-4V. The attack success rate is 45%!.


## Introduction

---

Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., 26% attack success rate against Bing Chat and 86\% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. 

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/VQA.png)


## Setup

1. (Optional) Update conda:

`conda update -n base -c defaults conda`

2. Create a conda environment :

`conda create -n universal_vlm_jailbreak_env python=3.11`

3. Activate the environment:

`conda activate universal_vlm_jailbreak_env`

4. Update pip in preparation of the next step.

`pip install --upgrade pip`

5. Manually install a few additional packages using pip that aren't/weren't available via conda:

`pip install joblib pandas matplotlib seaborn nvidia-htop`

6. Make sure to log in to W&B by running `wandb login`.

7. Obtain the git submodules: `git submodule update --init --recursive`

8. cd into `submodules/prismatic` and run `pip install -e .`.
9. Then follow their instructions:

`pip install packaging ninja`

`pip install flash-attn --no-build-isolation`

## Optimizing Jailbreaks

## Evaluating Jailbreaks

```
cd PerezAstraFellowship-Image-DAN-VLM-Attack
conda activate universal_vlm_jailbreak_env
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
wandb agent rylan/universal-vlm-jailbreak-eval/qrq79qj5
```


## Results



Our code is implemented based on [**MiniGPT4**](https://github.com/Vision-CAIR/MiniGPT-4) and [**AdversarialAttacks**](https://github.com/huanranchen/AdversarialAttacks).  Thanks them for supporting! 



