

from deepseek_vl.models.processing_vlm import VLChatProcessor


model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

decode_me = [380]
decoded = tokenizer.decode(decode_me)
print(decoded)
encode_me = "B"
encoded = tokenizer.encode(encode_me)
print(encoded)
