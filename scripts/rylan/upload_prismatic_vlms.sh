

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
  --model_id "llama2-chat+7b+clip" \
  --run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+clip+stage-finetune+x7"

python -u upload.py \
--model_id "llama2-chat+7b+siglip" \
--run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+siglip+stage-finetune+x7"


python -u upload.py \
--model_id "llama2-chat+7b+dinosiglip" \
--run_dir "/workspace/prismatic-vlms/runs/llama2-chat+7b+dinosiglip+stage-finetune+x7"



```python
from prismatic import (
    available_models,
    get_model_description,
    load,
)

model_str =

self.model = load(model_id_or_path=model_str)
```