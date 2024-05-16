from transformers import PretrainedConfig
from transformers import logging
from transformers import CONFIG_MAPPING

logger = logging.get_logger(__name__)


class Blip3VisionEncoderConfig(PretrainedConfig):
    model_type = "blip_3_vision_encoder"

    def __init__(
        self,
        model_name: str = "ViT-H-14-378-quickgelu",
        force_image_size: int = 378,
        **kwargs,
    ):
        self.model_name = model_name
        self.force_image_size = force_image_size
        super().__init__(**kwargs)


class Blip3VisionTokenizerConfig(PretrainedConfig):
    model_type = "blip_3_vision_tokenizer"

    def __init__(
        self,
        vis_feature_dim: int = 1280,
        lang_embedding_dim: int = 3072,
        num_vis_tokens: int = 128,
        image_aspect_ratio: str = "anyres",
        repeat_latents: bool = False,
        **kwargs,
    ):
        self.vis_feature_dim = vis_feature_dim
        self.lang_embedding_dim = lang_embedding_dim
        self.num_vis_tokens = num_vis_tokens
        self.image_aspect_ratio = image_aspect_ratio
        self.repeat_latents = repeat_latents
        super().__init__(**kwargs)


class Blip3Config(PretrainedConfig):
    model_type = "blip_3"

    def __init__(
        self,
        vision_encoder_config: dict = None,
        vision_tokenizer_config: dict = None,
        text_config: dict = None,
        **kwargs,
    ):

        if vision_encoder_config is None:
            vision_encoder_config = {
                "image_aspect_ratio": "anyres",
                "anyres_patch_sampling": True,
            }
            logger.info(
                "vision_encoder_config is None. initializing the Blip3VisionEncoderConfig with default values."
            )

        if vision_tokenizer_config is None:
            vision_tokenizer_config = {}
            logger.info(
                "vision_tokenizer_config is None. Initializing the Blip3VisionTokenizerConfig with default values."
            )

        if text_config is None:
            text_config = {
                "initial_tokenizer_len": 32012,
                "pad_token_id": 32011,
                "bos_token_id": 1,
                "eos_token_id": 32000,
                "vocab_size": 32064,
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
                "attention_dropout": 0.0,
                "hidden_act": "silu",
                "max_position_embeddings": 4096,
                "original_max_position_embeddings": 4096,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-05,
                "use_cache": True,
                "rope_theta": 10000.0,
                "rope_scaling": None,
                "sliding_window": 2047,
                "return_dict": True,
                "output_hidden_states": False,
                "output_attentions": False,
                "torchscript": False,
                "torch_dtype": "bfloat16",
                "use_bfloat16": False,
                "tf_legacy_loss": False,
                "pruned_heads": {},
                "tie_word_embeddings": False,
                "chunk_size_feed_forward": 0,
                "is_encoder_decoder": False,
                "is_decoder": False,
                "cross_attention_hidden_size": None,
                "add_cross_attention": False,
                "tie_encoder_decoder": False,
                "max_length": 20,
                "min_length": 0,
                "do_sample": False,
                "early_stopping": False,
                "num_beams": 1,
                "num_beam_groups": 1,
                "diversity_penalty": 0.0,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
                "typical_p": 1.0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "encoder_no_repeat_ngram_size": 0,
                "bad_words_ids": None,
                "num_return_sequences": 1,
                "output_scores": False,
                "return_dict_in_generate": False,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
                "remove_invalid_values": False,
                "exponential_decay_length_penalty": None,
                "suppress_tokens": None,
                "begin_suppress_tokens": None,
                "finetuning_task": None,
                "id2label": {0: "LABEL_0", 1: "LABEL_1"},
                "label2id": {"LABEL_0": 0, "LABEL_1": 1},
                "tokenizer_class": None,
                "prefix": None,
                "bos_token_id": 1,
                "pad_token_id": 32000,
                "eos_token_id": 32000,
                "sep_token_id": None,
                "decoder_start_token_id": None,
                "task_specific_params": None,
                "problem_type": None,
                "model_type": "phi3",
            }
            logger.info(
                "text_config is None. Initializing the text config with default values (`Phi3Config`)."
            )

        self.vision_encoder_config = Blip3VisionEncoderConfig(**vision_encoder_config)

        self.vision_tokenizer_config = Blip3VisionTokenizerConfig(
            **vision_tokenizer_config
        )

        text_model_type = (
            text_config["model_type"] if "model_type" in text_config else "phi3"
        )
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        for key in ["initial_tokenizer_len", "pad_token_id"]:
            if key not in self.text_config.to_dict():
                raise ValueError(f"The key `{key}` is missing in the text_config.")

        super().__init__(**kwargs)

    @classmethod
    def from_vision_encoder_vision_tokenizer_text_configs(
        cls,
        vision_encoder_config: Blip3VisionEncoderConfig,
        vision_tokenizer_config: Blip3VisionTokenizerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):

        return cls(
            vision_encoder_config=vision_encoder_config.to_dict(),
            vision_tokenizer_config=vision_tokenizer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )
