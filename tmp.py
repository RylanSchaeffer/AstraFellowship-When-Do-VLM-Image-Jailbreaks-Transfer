# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
#
# model_name = "liuhaotian/llava-v1.5-vicuna-7b"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
)
