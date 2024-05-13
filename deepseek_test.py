import time
import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",
        "images": ["images/trina/000.jpg"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]

conversation_2 = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image. and explain the process of the image.",
        "images": ["images/trina/000.jpg"]
    },
    {
        "role": "Assistant",
        "content": ""
    }
]


# load images and prepare for inputs
pil_images = load_pil_images(conversation)

time_start = time.time()
input_1 = vl_chat_processor.process_one(
            prompt=None, conversations=conversation, images=pil_images
)
time_end = time.time()
time_taken = time_end - time_start
print(f"Time taken: {time_taken} seconds")
input_2 = vl_chat_processor.process_one(
            prompt=None, conversations=conversation_2, images=pil_images
)
result = vl_chat_processor.batchify([input_1, input_2])
print(result)


# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**input_1)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

# answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# print(f"{prepare_inputs['sft_format'][0]}", answer)
