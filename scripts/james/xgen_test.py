from email.mime import image
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
from src.models.xgen import XgenVisionLanguageModel

os.environ["HF_HOME"] = "/workspace/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/huggingface_cache/hub"

torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dtype = torch.bfloat16
model = XgenVisionLanguageModel().to(device).to(dtype=dtype).eval()
tokenizer = model.tokenizer
model.disable_model_gradients()

image_path = f"images/trina/000.jpg"
pil_image = Image.open(image_path, mode="r")
width, height = pil_image.size
max_dim = max(width, height)
pad_width = (max_dim - width) // 2
pad_height = (max_dim - height) // 2
transform_pil_image = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.Resize(
            (512, 512)
        ),
        torchvision.transforms.v2.ToTensor(),  # This divides by 255.
    ]
)
transformed_image: torch.Tensor = transform_pil_image(pil_image)
# transformed_image: torch.Tensor = model.model.transformer.visual.image_transform(
#     pil_image
# )  # type: ignore
# print(f"Transformed to image: {image}")
image = transformed_image.unsqueeze(0)

response = model.generate(image=image, prompts=["What animal is in this picture?"])
print(response)


batch = model.convert_prompts_and_maybe_targets_to_input_ids_and_attention_mask(
    prompts=["What animal is this?\nA - Fish\nB - Cat\nC - Dog\nD - Whale"],
    targets=["C"],
)
loss = model.compute_loss(
    image=image,
    input_ids=batch["input_ids"].to(device=device),
    attention_mask=batch["attention_mask"].to(device=device),
    labels=batch["labels"].to(device=device),
)
print(f"Loss: {loss.item()}")
