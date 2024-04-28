

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def main():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    token = "D"
    print(tokenizer.encode(token, add_special_tokens=False))
    decode_this = torch.Tensor([360])
    print(tokenizer.decode(decode_this))
    # print(tokenizer.decode([319]))

main()