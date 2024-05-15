from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': url}, # Either a local path or an url
    {'text': '这是什么?'},
])
print(type(tokenizer))
print(query)