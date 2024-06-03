#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ========================== Mistral 7B Instruct v0.2 ==========================

## Mistral 7B Instruct v0.2 + CLIP
## https://wandb.ai/rylan/prismatic-vlm/runs/gtchsroj?nw=nwuserrylan
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "mistral-instruct-v0.2+7b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "mistral-v0.2-7b-instruct" \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"

## Mistral 7B Instruct v0.2 + SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "mistral-instruct-v0.2+7b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "mistral-v0.2-7b-instruct" \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"

## Mistral 7B Instruct v0.2 + DINOv2+SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "mistral-instruct-v0.2+7b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "mistral-v0.2-7b-instruct" \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"

# ========================== Gemma 8B Instruct ==========================

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
## Gemma Instruct 8B + CLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+8b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-8b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
#
## Gemma Instruct 8B + SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+8b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-8b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
#
## Gemma Instruct 8B + DINOv2+SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+8b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-8b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"


# ========================== Gemma 2B Instruct ==========================

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
## Gemma Instruct 2B + CLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+2b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-2b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
#
## Gemma Instruct 2B + SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+2b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-2b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
#
## Gemma Instruct 2B + DINOv2+SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "gemma-instruct+2b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "gemma-2b-instruct" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"


# ========================== Llama 2 7B ==========================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Llama 2 7B + CLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama2+7b+clip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "clip-vit-l-336px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"

# Llama 2 7B + SigLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama2+7b+siglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"

# Llama 2 7B + DINOv2SigLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama2+7b+dinosiglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama2-7b-pure" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"


# ========================== Llama 2 7B Chat ==========================

### Llama 2 Chat 7B + CLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama2-chat+7b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "llama2-7b-chat" \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"

## Llama 2 Chat 7B + SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama2-chat+7b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "llama2-7b-chat" \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
## Llama 2 Chat 7B + DINOv2SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama2-chat+7b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "llama2-7b-chat" \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"


# ========================== Llama 3 8B ==========================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Llama 3 8B + CLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama3+8b+clip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama3-8b-pure" \
  --model.vision_backbone_id "clip-vit-l-336px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"

# Llama 3 8B + SigLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama3+8b+siglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama3-8b-pure" \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"

# Llama 3 8B + DINOv2SigLIP
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "llama3+8b+dinosiglip" \
  --model.image_resize_strategy "letterbox" \
  --model.finetune_global_batch_size 128 \
  --model.finetune_per_device_batch_size 4 \
  --model.llm_backbone_id "llama3-8b-pure" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --wandb_entity "rylan" \
  --wandb_project "prismatic-vlm"


# ========================== Llama 3 8B Instruct ==========================

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
## Llama 3 Instruct 8B + CLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama3-instruct+8b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.llm_backbone_id "llama3-8b-instruct" \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
## Llama 3 Instruct 8B + SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama3-instruct+8b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.llm_backbone_id "llama3-8b-instruct" \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"

## Llama 3 Instruct 8B + DINOv2SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "llama3-instruct+8b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.finetune_global_batch_size 128 \
#  --model.finetune_per_device_batch_size 4 \
#  --model.llm_backbone_id "llama3-8b-instruct" \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"


# ========================== Phi 3 Instruct 4B ==========================

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
## Phi 3 Instruct 4B + CLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "phi-instruct-3+4b+clip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "phi-3-instruct-4b" \
#  --model.vision_backbone_id "clip-vit-l-336px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"


## Phi 3 Instruct 4B + SigLIP
# Note: I had a typo and called this model "phi-instruct-3+4b+siglipi" instead of "phi-instruct-3+4b+siglip"
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "phi-instruct-3+4b+siglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "phi-3-instruct-4b" \
#  --model.vision_backbone_id "siglip-vit-so400m-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
#
## Phi 3 Instruct 4B + DINOv2SigLIP
#torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
#  --model.type "one-stage+7b" \
#  --model.model_id "phi-instruct-3+4b+dinosiglip" \
#  --model.image_resize_strategy "letterbox" \
#  --model.llm_backbone_id "phi-3-instruct-4b" \
#  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
#  --wandb_entity "rylan" \
#  --wandb_project "prismatic-vlm"
