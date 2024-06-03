

#python -u upload.py \
#  --model_id "mistral-instruct-v0.2+7b+clip" \
#  --run_dir "/workspace/prismatic-vlms/runs/mistral-instruct-v0.2+7b+clip+stage-finetune+x7"

#python -u upload.py \
#  --model_id "mistral-instruct-v0.2+7b+siglip" \
#  --run_dir "/workspace/prismatic-vlms/runs/mistral-instruct-v0.2+7b+siglip+stage-finetune+x7"

#python -u upload.py \
#  --model_id "mistral-instruct-v0.2+7b+dinosiglip" \
#  --run_dir "/workspace/prismatic-vlms/runs/mistral-instruct-v0.2+7b+dinosiglip+stage-finetune+x7"


python -u upload.py \
  --model_id "llama2+7b+clip" \
  --run_dir "/workspace/prismatic-vlms/runs/llama2+7b+clip+stage-finetune+x7"


python -u upload.py \
--model_id "llama2+7b+siglip" \
--run_dir "/workspace/prismatic-vlms/runs/llama2+7b+siglip+stage-finetune+x7"


python -u upload.py \
--model_id "llama2+7b+dinosiglip" \
--run_dir "/workspace/prismatic-vlms/runs/llama2+7b+dinosiglip+stage-finetune+x7"


#python -u upload.py \
#  --model_id "llama2-chat+7b+clip" \
#  --run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+clip+stage-finetune+x7"


#python -u upload.py \
#--model_id "llama2-chat+7b+siglip" \
#--run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+siglip+stage-finetune+x7"


#python -u upload.py \
#--model_id "llama2-chat+7b+dinosiglip" \
#--run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+dinosiglip+stage-finetune+x7"


#python -u upload.py \
#--model_id "phi-instruct-3+4b+clip" \
#--run_dir "/workspace/prismatic-vlms/runs/phi-instruct-3+4b+clip+stage-finetune+x7"

#python -u upload.py \
#--model_id "phi-instruct-3+4b+siglip" \
#--run_dir "/workspace/prismatic-vlms/runs/phi-instruct-3+4b+siglip+stage-finetune+x7"

#python -u upload.py \
#--model_id "phi-instruct-3+4b+dinosiglip" \
#--run_dir "/workspace/prismatic-vlms/runs/phi-instruct-3+4b+dinosiglip+stage-finetune+x7"


python -u upload.py \
--model_id "llama3+8b+clip" \
--run_dir "/workspace/prismatic-vlms/runs/llama3+8b+clip+stage-finetune+x7"


python -u upload.py \
--model_id "llama3+8b+siglip" \
--run_dir "/workspace/prismatic-vlms/runs/llama3+8b+siglip+stage-finetune+x7"


python -u upload.py \
--model_id "llama3+8b+clip" \
--run_dir "/workspace/prismatic-vlms/runs/llama3+8b+dinosiglip+stage-finetune+x7"


#python -u upload.py \
#--model_id "llama3-instruct+8b+clip" \
#--run_dir "runs/llama3-instruct+8b+clip+stage-finetune+x7"


#python -u upload.py \
#--model_id "llama3-instruct+8b+siglip" \
#--run_dir "runs/llama3-instruct+8b+siglip+stage-finetune+x7"


#python -u upload.py \
#--model_id "llama3-instruct+8b+dinosiglip" \
#--run_dir "runs/llama3-instruct+8b+dinosiglip+stage-finetune+x7"


#python -u upload.py \
#--model_id "gemma-instruct+8b+clip" \
#--run_dir "runs/gemma-instruct+8b+clip+stage-finetune+x7"


#python -u upload.py \
#--model_id "gemma-instruct+8b+siglip" \
#--run_dir "runs/gemma-instruct+8b+siglip+stage-finetune+x7"

#python -u upload.py \
#--model_id "gemma-instruct+8b+dinosiglip" \
#--run_dir "runs/gemma-instruct+8b+dinosiglip+stage-finetune+x7"

#python -u upload.py \
#--model_id "gemma-instruct+2b+clip" \
#--run_dir "runs/gemma-instruct+2b+clip+stage-finetune+x7"
#
#
#python -u upload.py \
#--model_id "gemma-instruct+2b+siglip" \
#--run_dir "runs/gemma-instruct+2b+siglip+stage-finetune+x7"
#
#python -u upload.py \
#--model_id "gemma-instruct+2b+dinosiglip" \
#--run_dir "runs/gemma-instruct+2b+dinosiglip+stage-finetune+x7"


```python
from prismatic import (
    available_models,
    get_model_description,
    load,
)

model_str =

self.model = load(model_id_or_path=model_str)
```