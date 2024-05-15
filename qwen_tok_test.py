from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# pad_token = tokenizer.eos_token_id
# pad_token_id = tokenizer.pad_token_id
pad_token_id = 55
"""
n theory, the model never sees or computes this token, so you may use any known token. But to be safe, we limit the value of special tokens specified in the initialization of the tokenizer to the known special tokens. You may specify special tokens in fine-tuning or in any other frameworks that necessitate them like this

"""


print(tokenizer.decode([pad_token_id]))

url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': url}, # Either a local path or an url
    {'text': '这是什么?'},
])
# print(type(tokenizer))
# print(query)