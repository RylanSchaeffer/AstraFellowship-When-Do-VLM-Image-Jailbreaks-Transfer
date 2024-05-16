from transformers import PreTrainedModel, AutoModelForCausalLM
import torch
import open_clip
from typing import List, Optional, Tuple, Union
from .utils import check_embedding_fns
from .vlm import InstructPerceiverResampler, KosmosInstruct
from .configuration_blip_3 import (
    Blip3VisionEncoderConfig,
    Blip3VisionTokenizerConfig,
    Blip3Config,
)


class Blip3VisionEncoder(PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = Blip3VisionEncoderConfig

    def __init__(self, config: Blip3VisionEncoderConfig):
        super().__init__(config)
        if config.model_name != "ViT-H-14-378-quickgelu":
            raise ValueError(
                f"Unsupported model {config.model_name}. New vision models will be added soon."
            )
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=config.model_name, force_image_size=config.force_image_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # assert pixel_values.ndim == 4, f"Expected 4D tensor (bs, c, h, w), got {pixel_values.ndim}"
        return self.model.encode_image(pixel_values)


# vision tokenizer
class Blip3VisionTokenizer(PreTrainedModel):
    config_class = Blip3VisionTokenizerConfig

    def __init__(self, config: Blip3VisionTokenizerConfig):
        super().__init__(config)
        self.model = InstructPerceiverResampler(
            dim_llm=config.lang_embedding_dim,
            dim=config.vis_feature_dim,
            dim_inner=config.lang_embedding_dim,
            num_latents=config.num_vis_tokens,
            repeat_latents=config.repeat_latents,
        )

    def forward(self, vision_features: torch.Tensor, vision_attn_masks: torch.Tensor):
        return self.model(vision_features, vision_attn_masks)


# Blip3 model
class Blip3ModelForConditionalGeneration(PreTrainedModel):
    config_class = Blip3Config

    def __init__(self, config: Blip3Config):
        super().__init__(config)

        # vision encoder initialization
        vision_encoder = Blip3VisionEncoder(config.vision_encoder_config).model
        vision_encoder.visual.output_tokens = True
        vision_encoder = vision_encoder.visual

        # language model initialization
        language_model = AutoModelForCausalLM.from_config(config.text_config)
        check_embedding_fns(language_model)
        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]

        # vision tokenizer initialization
        if (
            config.vision_tokenizer_config.lang_embedding_dim
            != language_model.get_input_embeddings().weight.shape[1]
        ):
            overwrite = language_model.get_input_embeddings().weight.shape[1]
            config.vision_tokenizer_config.lang_embedding_dim = overwrite
            print(
                f"Warning: The language embedding dimension in the vision tokenizer config is different from the language model's embedding dimension. Overwriting the language embedding dimension in the vision tokenizer config to {overwrite}."
            )

        vision_tokenizer = Blip3VisionTokenizer(config.vision_tokenizer_config).model

        self.vlm = KosmosInstruct(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=language_model,
            initial_tokenizer_len=config.text_config.initial_tokenizer_len,
            pad_token_id=config.text_config.pad_token_id,
            image_aspect_ratio=config.vision_encoder_config.image_aspect_ratio,
            anyres_patch_sampling=config.vision_encoder_config.anyres_patch_sampling,
        )
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        self.vlm: KosmosInstruct = self.vlm.eval()
        return self.vlm.generate(
            vision_x=pixel_values,
            lang_x=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def update_special_tokens(self, tokenizer):
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(self.vlm.special_tokens.values())}
        )
        self.vlm.lang_model.config.vocab_size = len(tokenizer)
        self.vlm.set_special_token_ids(
            {
                v: tokenizer.convert_tokens_to_ids(v)
                for v in self.vlm.special_tokens.values()
            }
        )
        return tokenizer
