

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def main():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    token = "A"
    print(tokenizer.encode(token))
    decode_this = torch.Tensor([1, 29871, 319])
    print(tokenizer.decode(decode_this))
    print(tokenizer.decode([319]))

main()