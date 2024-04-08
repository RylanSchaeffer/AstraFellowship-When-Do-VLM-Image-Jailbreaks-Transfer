import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaGuardEvaluator:
    def __init__(self, device: int, dtype: torch.dtype = torch.bfloat16):
        # See: https://huggingface.co/meta-llama/LlamaGuard-7b
        self.model_id = "meta-llama/LlamaGuard-7b"
        self.device = f"cuda:{str(device)}"
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.dtype, device_map=self.device
        )

    def moderate(self, prompt: str, generated_output: str) -> str:
        chat = [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": generated_output,
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0
        )
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
