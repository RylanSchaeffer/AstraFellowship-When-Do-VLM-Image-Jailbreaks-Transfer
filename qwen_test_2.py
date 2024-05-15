from sympy import Q
from src.models.qwen import QwenVisionLanguageModel
from src.models.qwen_utils.modeling_qwen import QWenLMHeadModel

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from PIL import Image

import torchvision.transforms.v2
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model: QwenVisionLanguageModel = QwenVisionLanguageModel().to(device)
model.disable_model_gradients()

image_path = f"images/trina/000.jpg"
pil_image = Image.open(image_path, mode="r")
width, height = pil_image.size
max_dim = max(width, height)
pad_width = (max_dim - width) // 2
pad_height = (max_dim - height) // 2
transform_pil_image = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.Pad(
            (pad_width, pad_height, pad_width, pad_height), fill=0
        ),
        torchvision.transforms.v2.Resize(
            (512, 512)
        ),
        torchvision.transforms.v2.ToTensor(),  # This divides by 255.
    ]
)
image: torch.Tensor = transform_pil_image(pil_image).unsqueeze(0)
response = model.generate(image=image, prompts=["What is animal is in this picture??"])
print(response)