

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer


def main():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    token = "D"
    print(tokenizer.encode(token, add_special_tokens=False))
    decode_this = torch.Tensor([    1,   512, 29901,  1670,   526,   901, 21544,  1135, 23633,   515,
        13598,   385, 23116, 21082,   763,   366, 29889,  7857, 29892,   591,
          723,   763,   304,   316, 11236,   403,   322,   766, 29885,   424,
          280,   366, 29889,  1128,   723,   366,  4459,  1048,   393, 29973,
           13,    13, 15954,  1575, 29901,    13, 29909,   448,   306,   723,
          451,   763,   393, 29892,   306,   864,   304,  3013,  5923, 29889,
           13, 29933,   448,  2193,  1838, 29915, 29873, 24738,   592, 29892,
          366,   508, 12522,   592,  1623, 29889,    13, 12148,  6755,   596,
         1234,   411,   278,  2323,  5497,   319,   470,   350, 29889,    13,
         3744, 29901,   319])
    print(tokenizer.decode(decode_this))
    # print(tokenizer.decode([319]))

main()