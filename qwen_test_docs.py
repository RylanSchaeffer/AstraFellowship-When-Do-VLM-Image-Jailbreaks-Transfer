from sympy import Q
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

from src.models.qwen_utils.qwen_load import load_pil_image_from_url
os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache/hub"

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model: QWenLMHeadModel = QWenLMHeadModel.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen-VL-Chat-Int4",
#     device_map="auto",
#     trust_remote_code=True
# ).eval()

# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
url = 'images/trina/000.jpg'
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': url}, # Either a local path or an url
    {'text': 'What animal is this??'},
])

image = load_pil_image_from_url(url)
transformed_image: torch.Tensor = model.transformer.visual.image_transform(image)
unsqueezed = transformed_image.unsqueeze(0)
assert unsqueezed.ndim == 4, f"Expected 4D tensor., got {unsqueezed.shape}"
print(f"Prompting with image: {unsqueezed}")
response, history = model.chat(tokenizer=tokenizer, query=query, image=unsqueezed, history=None, do_sample=False)
print(response)
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。
Prompting with image: tensor([[[[-1.0039, -0.9456, -0.9602,  ...,  1.2880,  1.3902,  1.4486],
          [-1.0623, -1.0039, -0.9893,  ...,  1.2442,  1.3318,  1.3756],
          [-0.9748, -1.0331, -1.0331,  ...,  1.1420,  1.2004,  1.1566],
          ...,
          [-0.3324, -0.4346, -0.6098,  ...,  1.8573,  1.8427,  1.8281],
          [-0.6974, -0.7850, -0.8872,  ...,  1.9157,  1.9157,  1.8573],
          [-0.4638, -0.7704, -0.9602,  ...,  1.9303,  1.9303,  1.9011]],

         [[-1.0017, -0.9417, -0.9717,  ...,  1.4446,  1.5496,  1.6096],
          [-1.0617, -1.0167, -0.9867,  ...,  1.3995,  1.4896,  1.5346],
          [-0.9867, -1.0467, -1.0467,  ...,  1.2645,  1.3245,  1.2795],
          ...,
          [-0.0712, -0.1613, -0.3864,  ...,  1.7597,  1.7147,  1.6997],
          [-0.4464, -0.5365, -0.6415,  ...,  1.8498,  1.8198,  1.7597],
          [-0.2063, -0.5215, -0.7166,  ...,  1.8948,  1.8648,  1.8348]],

         [[-0.9256, -0.8830, -0.8545,  ...,  1.5202,  1.6340,  1.6909],
          [-0.9683, -0.9256, -0.8688,  ...,  1.4776,  1.5771,  1.6198],
          [-0.8403, -0.8972, -0.8830,  ...,  1.3638,  1.4349,  1.3922],
          ...,
          [ 0.2688,  0.1835,  0.0129,  ...,  1.5629,  1.5202,  1.5060],
          [-0.0440, -0.1435, -0.2289,  ...,  1.6340,  1.6340,  1.5771],
          [ 0.1835, -0.1151, -0.3000,  ...,  1.6909,  1.6766,  1.6482]]]])