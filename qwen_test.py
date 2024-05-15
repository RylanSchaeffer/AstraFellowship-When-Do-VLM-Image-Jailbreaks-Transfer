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
url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': url}, # Either a local path or an url
    {'text': '这是什么?'},
])

image = load_pil_image_from_url(url)
transformed_image: torch.Tensor = model.transformer.visual.image_transform(image)
response, history = model.chat(tokenizer=tokenizer, query=query, image=transformed_image, history=None)
print(response)
# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。
