import torch
from torch import einsum, nn
from einops import rearrange, repeat
from einops_exts import rearrange_many
from einops import rearrange
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from transformers import CLIPVisionModel
import transformers

from .utils import (
    num_params,
    getattr_recursive,
    stack_with_padding,
    get_anyres_image_grid_shape,
    unpad_image,
)


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat(
            (x, latents), dim=-2
        )  # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat(
                (
                    vision_attn_masks,
                    torch.ones(
                        (latents.shape[0], latents.shape[-2]),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                ),
                dim=-1,
            )
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros(
                (q.size(0), 1, 1, q.size(-2), k.size(-2)),
                dtype=q.dtype,
                device=q.device,
            )
            vision_attn_masks = repeat(
                vision_attn_masks, "b n -> b 1 1 l n", l=q.size(-2)
            )
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class InstructPerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim_llm,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=64,
        repeat_latents=False,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection

        # Text embedding projection.
        # self.text_projection = nn.Linear(dim_llm, dim)
        modules = [nn.Linear(dim_llm, dim)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(dim, dim))
        self.text_projection = nn.Sequential(*modules)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.repeat_latents = repeat_latents
        # positional embeddings
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    # TODO: write a new forward function that takes in text input and append them to the query tokens.
    def forward(self, x, text_embeds=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        # FIXME: extending query tokens proportional to the vision sequence length. Hard-coded as dfn5b token_len=729.
        if self.repeat_latents:
            r = v // 729  # Repeat the query tokens for r times.
            latents = repeat(self.latents, "n d -> (n repeat) d", repeat=r)
        else:
            latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        # latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        # Append text embedding.
        if exists(text_embeds):
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds[:, None, :, :]
            latents = torch.cat(
                (latents, text_embeds), dim=2
            )  # FIXME: check latents shape.

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # Truncate latents to only keep query tokens.
        if exists(text_embeds):
            latents = latents[:, :, : self.latents.shape[0], :]

        if exists(self.projection):
            return self.projection(self.norm(latents))
        else:
            return self.norm(latents)


class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0,
    then it will create `num_additional_embeddings` additional parameters that are always trained. If
    `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        max_original_id: int,
        num_additional_embeddings: int = 0,
        _weight: torch.Tensor = None,
        num_original_embeddings: int = None,
        embedding_dim: int = None,
        partially_freeze=True,
        device=None,
        dtype=None,
        pad_token_id=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`):
                The largest token id that should be embedded using the regular embedding (regular `weight`).
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal self.weight.shape[0]
            num_additional_embeddings (`int`):
                Number of additional tokens to initialize an Embedding matrix for (`additional_weight`).
            _weight (`torch.Tensor`, *optional*, defaults to `None`): The regular weight tensor.
                If provided, this sets the `num_original_embeddings` and `embedding_dim` parameters.
            num_original_embeddings (`int`):
                self.weight.shape[0]
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `True`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # validate args
        if pad_token_id is not None and pad_token_id > max_original_id:
            raise ValueError(
                f"pad_token_id must be <= max_original_id. Got {pad_token_id} and {max_original_id}."
                + "If the original tokenizer does not have a pad_token_id, use pad_token_id=None."
            )
        if _weight is not None:
            assert (num_original_embeddings is None) or (
                _weight.shape[0] == num_original_embeddings
            ), f"num_original_embeddings={num_original_embeddings} but _weight.shape[0]={_weight.shape[0]}"
            assert (embedding_dim is None) or (
                _weight.shape[1] == embedding_dim
            ), f"embedding_dim={embedding_dim} but _weight.shape[1]={_weight.shape[1]}"
            num_original_embeddings = _weight.shape[0]
            embedding_dim = _weight.shape[1]
        else:
            assert (
                num_original_embeddings is not None
            ), "num_original_embeddings must be provided if _weight is not provided"
            assert (
                embedding_dim is not None
            ), "embedding_dim must be provided if _weight is not provided"

        super().__init__(
            num_embeddings=num_original_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=pad_token_id,
            _weight=_weight,
        )
        self.max_original_id = max_original_id
        self.padding_idx = pad_token_id
        self.num_additional_embeddings = num_additional_embeddings
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        self.additional_embedding.requires_grad_(require_additional_grad)

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
        embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids > self.max_original_id)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(
            input_ids_additional_vocab - self.max_original_id - 1
        )

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        return "num_original_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.max_original_id + 1,
            self.num_additional_embeddings,
            self.embedding_dim,
            (not self.weight.requires_grad),
        )


class DecoupledLinear(nn.Linear):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `additional_out_features` > 0,
    then it will create `additional_out_features * in_features` additional parameters that are always trained. If
    `additional_out_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        max_original_id: int,
        additional_out_features: int = 0,
        _weight: torch.Tensor = None,
        _bias: torch.Tensor = None,
        in_features: int = None,
        original_out_features: int = None,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            max_original_id (`int`): The largest token id that should be extracted from the regular weight.
                This is usually len(tokenizer) - 1 before additional tokens are added.
                Note that this may not equal original_out_features - 1
            _weight: torch.Tensor, *optional*, defaults to `None`. The regular weight tensor.
                If provided, this sets the `in_features` and `original_out_features` parameters.
            _bias: torch.Tensor, *optional*, defaults to `None`. The regular bias tensor.
            in_features: int. Input hidden size.
            original_out_features: int. Original out_features of the language model's get_output_embeddings() function.
            additional_out_features: int. Number of additional trainable dimensions.
            bias: bool. Whether to include a bias term.
            partially_freeze: bool, *optional*, defaults to `True`): If `True`, the regular `weight` will be frozen.
        """
        # argument validation
        if _weight is not None:
            assert (_weight.shape[0] == original_out_features) or (
                original_out_features is None
            ), f"original_out_features={original_out_features} but _weight.shape[0]={_weight.shape[0]}"
            assert (_weight.shape[1] == in_features) or (
                in_features is None
            ), f"in_features={in_features} but _weight.shape[1]={_weight.shape[1]}"
            in_features = _weight.shape[1]
            original_out_features = _weight.shape[0]
        else:
            assert (
                in_features is not None
            ), "in_features must be provided if _weight is not provided"
            assert (
                original_out_features is not None
            ), "original_out_features must be provided if _weight is not provided"

        if _bias is not None:
            assert bias is True, "bias must be True if _bias is provided"

        # initialize original linear
        super().__init__(in_features, original_out_features, bias, device, dtype)

        # set weight and bias manually
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        if _bias is not None:
            self.bias = nn.Parameter(_bias)

        self.in_features = in_features
        self.original_out_features = original_out_features
        self.max_original_id = max_original_id

        # initialize additional linear
        self.additional_out_features = additional_out_features
        self.has_bias = bias
        if additional_out_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=additional_out_features,
                bias=self.has_bias,
                device=device,
                dtype=dtype,
            )
        self.set_requires_grad(
            require_regular_grad=not partially_freeze, require_additional_grad=True
        )

    def set_requires_grad(self, require_regular_grad, require_additional_grad):
        """
        Helper function to separately set the requires_grad flag for the regular weight and the additional weight.
        """
        self.weight.requires_grad_(require_regular_grad)
        if self.has_bias:
            self.bias.requires_grad_(require_regular_grad)
        self.additional_fc.requires_grad_(require_additional_grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight, self.bias)
        output = output[..., : self.max_original_id + 1]

        if self.additional_out_features > 0:
            additional_features = F.linear(
                input, self.additional_fc.weight, self.additional_fc.bias
            )
            output = torch.cat((output, additional_features), -1)
        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, additional_out_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.max_original_id + 1,
            self.additional_out_features,
            self.bias is not None,
            (not self.weight.requires_grad or not self.bias.requires_grad),
        )


class VLM(nn.Module):
    """
    Generic vision-language model (VLM) class.
    A VLM consists of four components:
        1. A vision encoder that extracts features from pixels, e.g. CLIP
            input: (B, T_img, F, C, H, W)
            output: (B, T_img, F, v, d)
        2. A vision tokenizer that converts these features to visual token-like embeddings, e.g. Perceiver, or a linear projection head
            input: (B, T_img, F, v, d)
            output: (B, T_img, n, d)
        3. A fusion method that allows the language model to attend to these tokens, e.g. cross-attention, or placing the tokens directly in the language model's input sequence
        4. A language model
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): e.g. CLIP
            vision_tokenizer (nn.Module): e.g. PerceiverResampler
            lang_model (nn.Module): e.g. MPT
            initial_tokenizer_len (int): size of the original tokenizer vocab
            pad_token_id (int): id of the pad token
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()

        # save dimension information
        self.lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        if hasattr(lang_model.config, "d_model"):
            self.lang_hidden_dim = lang_model.config.d_model  # mpt uses d_model
        else:
            self.lang_hidden_dim = lang_model.config.hidden_size
        self.vis_embedding_dim = vision_tokenizer.dim_media
        self.num_tokens_per_vis = vision_tokenizer.num_tokens_per_media

        # core components
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        self.lang_model = lang_model

        # lm embeddings
        self.pad_token_id = pad_token_id
        self.initial_tokenizer_len = initial_tokenizer_len
        input_embeds = DecoupledEmbedding(
            max_original_id=initial_tokenizer_len - 1,
            num_additional_embeddings=len(self.special_tokens),
            _weight=self.lang_model.get_input_embeddings().weight,
            pad_token_id=self.pad_token_id,
        )
        if hasattr(input_embeds, "additional_embedding"):
            input_embeds.additional_embedding.weight.data.normal_(
                mean=0.0,
                std=(
                    self.lang_model.config.initializer_range
                    if hasattr(self.lang_model.config, "initializer_range")
                    else 0.02
                ),
            )
        self.lang_model.set_input_embeddings(input_embeds)

        out_embeds = DecoupledLinear(
            max_original_id=initial_tokenizer_len - 1,
            additional_out_features=len(self.special_tokens),
            _weight=self.lang_model.get_output_embeddings().weight,
            _bias=(
                self.lang_model.get_output_embeddings().bias
                if hasattr(self.lang_model.get_output_embeddings(), "bias")
                else None
            ),
        )
        if hasattr(out_embeds, "additional_fc"):
            out_embeds.additional_fc.weight.data.normal_(
                mean=0.0,
                std=(
                    self.lang_model.config.initializer_range
                    if hasattr(self.lang_model.config, "initializer_range")
                    else 0.02
                ),
            )
        self.lang_model.set_output_embeddings(out_embeds)

        # gradient checkpointing
        self.vision_tokenizer._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def _encode_vision_x_anyres(self, samples, device):
        image_raw = samples[
            "image"
        ]  # list of patch list in of shape [1, N_patch, C, H, W]
        image_sizes = samples["image_size"]

        # concate list of patches into one big patch for any res encoding.
        images = [x.squeeze(0) for x in image_raw]  # [N_patch, C, H, W]
        image = torch.cat(images, dim=0)  # [\sum{B}{N_patch_i}, C, H, W]
        image = image.to(device)

        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                image_embeds = self.vision_encoder.trunk.forward_features(image)
            elif self.vision_encoder.__class__.__name__ == "CLIPVisionModel":
                image_embeds = self.vision_encoder(image).last_hidden_state
            else:
                image_embeds = self.vision_encoder(image)[1]  # OpenCLIP returns tuples

        if isinstance(self.vision_encoder, CLIPVisionModel):
            base_img_size = self.vision_encoder.config.image_size
        else:
            base_img_size = self.vision_encoder.image_size[0]

        if self.vision_encoder.__class__.__name__ == "TimmModel":
            grid_size = self.vision_encoder.trunk.patch_embed.grid_size
        elif self.vision_encoder.__class__.__name__ == "CLIPVisionModel":
            grid_size_base = (
                self.vision_encoder.config.image_size
                // self.vision_encoder.config.patch_size
            )
            grid_size = (grid_size_base, grid_size_base)
        else:
            grid_size = self.vision_encoder.grid_size
        height, width = grid_size

        if not image_embeds.shape[1] == height * width:
            assert (
                image_embeds.shape[1] == height * width + 1
            )  # For vision encoders that has [CLS] token.
            image_embeds = image_embeds[:, 1:, :]  # Drop the cls token for each patch.
        n_vis_token_per_patch = image_embeds.shape[1]

        # Split encoded patches and merge patch features
        # 1. Get the raw sizes from samples, and split the image embeds [\sum_{B}(N_patch_i), N_tok(16*16), C]
        split_sizes = [image.shape[0] for image in images]
        image_embeds = torch.split(image_embeds, split_sizes, dim=0)
        # 2. For each image (consist of a list of patches), merge the patches spatially (of shape [C, n_patch_height, n_patch_width])
        new_image_embeds = []
        patch_attn_masks = []
        max_n_img_token = -1
        for idx, patch_embeds in enumerate(image_embeds):
            if patch_embeds.shape[0] > 1:
                # 3. Flatten the patch features and get [C, n_patch_height * (n_patch_width+1)]
                base_patch_embeds = patch_embeds[
                    0
                ]  # TODO: prepend the CLS token for th base patch embeds (of the resized entire image).
                patch_embeds = patch_embeds[1:]

                assert height * width == base_patch_embeds.shape[0]

                num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                    image_sizes[idx],
                    [
                        [base_img_size, base_img_size * 2],
                        [base_img_size * 2, base_img_size],
                        [base_img_size * 2, base_img_size * 2],
                        [base_img_size * 3, base_img_size],
                        [base_img_size, base_img_size * 3],
                    ],
                    base_img_size,
                )  # Hardcoded grid_pinpoints.
                patch_embeds = patch_embeds.view(
                    num_patch_height, num_patch_width, height, width, -1
                )

                patch_embeds = patch_embeds.permute(4, 0, 2, 1, 3).contiguous()
                patch_embeds = patch_embeds.flatten(1, 2).flatten(2, 3)
                # TODO: add an option that return masked patch_embeds instead of trimmed.
                patch_embeds, patch_attn_mask = unpad_image(
                    patch_embeds, image_sizes[idx], self.anyres_patch_sampling
                )
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat(
                        (
                            patch_embeds,
                            self.image_newline[:, None, None].expand(
                                *patch_embeds.shape[:-1], 1
                            ),
                        ),
                        dim=-1,
                    )
                if self.anyres_patch_sampling:
                    patch_embeds = patch_embeds.view(
                        -1, num_patch_height, num_patch_width, height * width
                    )
                    patch_embeds = patch_embeds.flatten(1, 2).permute(1, 2, 0)
                    assert patch_attn_mask is not None
                    patch_attn_mask = patch_attn_mask.view(
                        num_patch_height, num_patch_width, height * width
                    )
                    patch_attn_mask = patch_attn_mask.flatten(0, 1)
                    patch_embeds = torch.cat(
                        (base_patch_embeds.unsqueeze(0), patch_embeds), dim=0
                    )
                    patch_attn_mask = torch.cat(
                        (
                            torch.ones(
                                n_vis_token_per_patch, device=patch_embeds.device
                            ).unsqueeze(0),
                            patch_attn_mask,
                        ),
                        dim=0,
                    )
                else:
                    patch_embeds = patch_embeds.flatten(1, 2).transpose(0, 1)
                    patch_embeds = torch.cat((base_patch_embeds, patch_embeds), dim=0)
            else:
                patch_embeds = (
                    patch_embeds[0].unsqueeze(0)
                    if self.anyres_patch_sampling
                    else patch_embeds[0]
                )
                patch_attn_mask = (
                    torch.ones(
                        n_vis_token_per_patch, device=patch_embeds.device
                    ).unsqueeze(0)
                    if self.anyres_patch_sampling
                    else None
                )
                if hasattr(self, "image_newline"):
                    patch_embeds = torch.cat(
                        (patch_embeds, self.image_newline[None]), dim=0
                    )
            if not self.anyres_patch_sampling:
                max_n_img_token = max(patch_embeds.shape[0], max_n_img_token)

            new_image_embeds.append(patch_embeds)
            patch_attn_masks.append(patch_attn_mask)

        if self.anyres_patch_sampling:
            # Return individual patches for independent token downsampling.
            return new_image_embeds, patch_attn_masks

        # 4. Pad and concat the list of image_embeds [N_tok_i, C] together into a batch. Also modify the query attention mask.
        image_embeds = []
        image_atts = []
        for image_embed in new_image_embeds:
            n_img_token = image_embed.shape[0]
            img_attn = torch.ones(
                (max_n_img_token), dtype=torch.long, device=image_embed.device
            )
            if n_img_token < max_n_img_token:
                padded_embed = torch.zeros(
                    (max_n_img_token, image_embed.shape[-1]),
                    dtype=image_embed.dtype,
                    device=image_embed.device,
                )
                padded_embed[:n_img_token, :] = image_embed
                img_attn[n_img_token:] = 0  # Mask out the padded entries.
            else:
                padded_embed = image_embed
            image_embeds.append(padded_embed)
            image_atts.append(img_attn)
        image_embeds = torch.stack(
            image_embeds, dim=0
        )  # Shape [B, N_tok_longest, C_dim]
        image_atts = torch.stack(image_atts, dim=0)  # Shape [B, N_tok_longest, C_dim]
        # TODO: reshape image_embeds and image_atts to "b T F v d"
        image_embeds = image_embeds[:, None, None, :, :]
        # image_atts = image_atts[:, None, None, :, :]

        return image_embeds, image_atts

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            if self.vision_encoder.__class__.__name__ == "TimmModel":
                vision_x = self.vision_encoder.trunk.forward_features(vision_x)
            elif self.vision_encoder.__class__.__name__ == "CLIPVisionModel":
                vision_x = self.vision_encoder(vision_x).last_hidden_state
            else:
                vision_x = self.vision_encoder(vision_x)[1]  # OpenCLIP returns tuples
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        return vision_x

    def _concat_vision_cache(
        self, lang_x, vision_tokens, past_vision_tokens, past_media_locations, use_cache
    ):
        """
        Helper function to include the past vision tokens and past media locations in the output.
        """
        if use_cache:
            if past_media_locations is not None and past_vision_tokens is not None:
                if vision_tokens is not None:
                    updated_vision_tokens = torch.cat(
                        [
                            past_vision_tokens,
                            vision_tokens,
                        ],
                        dim=1,
                    )
                else:
                    updated_vision_tokens = past_vision_tokens
                updated_media_locations = torch.cat(
                    [
                        past_media_locations,
                        lang_x == self.media_token_id,
                    ],
                    dim=1,
                )
            else:
                updated_vision_tokens = vision_tokens
                updated_media_locations = lang_x == self.media_token_id

        else:
            updated_vision_tokens = None
            updated_media_locations = None

        return updated_vision_tokens, updated_media_locations

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        if vision_x is not None:
            vision_features = self._encode_vision_x(vision_x=vision_x)
            vision_tokens = self.vision_tokenizer(vision_features)
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        output = self.lang_model.generate(
            **new_inputs,
            past_key_values=past_key_values,
            num_beams=num_beams,
            use_cache=True,
            **kwargs,
        )
        self._post_forward_hook()
        return output

    @property
    def num_trainable_params(self):
        """Print the number of trainable parameters"""
        return num_params(self, filter_to_trainable=True)

    def set_trainable(self):
        """
        Freeze appropriate parameters in the model.
        """
        raise NotImplementedError

    def group_params_by_weight_decay(self):
        """
        Return a tuple of (params to optimize w/ weight decay, params to optimize w/o weight decay)
        """
        params_with_wd, params_without_wd = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if self._should_apply_weight_decay(n):
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
        return params_with_wd, params_without_wd

    def _should_apply_weight_decay(self, parameter_name):
        """
        Return whether weight decay should be applied to a parameter.
        """
        raise NotImplementedError

    @property
    def special_tokens(self):
        """
        Returns a dict mapping from the attribute name of a special token to its string format,
         e.g. "media_token": "<image>"
        """
        assert (
            "media_token" in self._special_tokens
        ), "VLMs need to request that the tokenizer add a media_token and call set_special_token_ids to set self.media_token_id"
        return self._special_tokens

    @property
    def special_token_ids(self):
        """
        Returns a list of the special token ids
        """
        return [getattr(self, f"{att_name}_id") for att_name in self.special_tokens]

    def set_special_token_ids(self, string_to_ids):
        """
        Args:
            string_to_ids (dict): mapping from token string to id
        """
        assert set(self.special_tokens.values()).issubset(set(string_to_ids.keys()))
        for att_name, token_str in self.special_tokens.items():
            token_id = string_to_ids[token_str]
            setattr(self, f"{att_name}_id", token_id)
            setattr(self.lang_model, f"{att_name}_id", token_id)

    def init_gradient_checkpointing(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointWrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        from functools import partial

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            self,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, CheckpointWrapper),
        )


@dataclass
class VLMOutputWithPast(CausalLMOutputWithPast):
    """
    VLMOutputWithPast is a wrapper around CausalLMOutputWithPast that adds the following attributes:
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
    """

    past_media_locations: Optional[torch.Tensor] = None
    past_vision_tokens: Optional[torch.Tensor] = None


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class VLMWithLanguageStream(VLM):
    """
    VLM that fuses modalities by inserting vision tokens directly into the language stream.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.decoder_layers_attr_name = decoder_layers_attr_name
        if decoder_layers_attr_name is not None:
            for block in getattr_recursive(
                self.lang_model, self.decoder_layers_attr_name
            ):
                block._use_gradient_checkpointing = gradient_checkpointing

    def _prepare_inputs_for_forward(
        self,
        vision_tokens: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        past_key_values=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        past_media_locations: torch.Tensor = None,
        past_vision_tokens: torch.Tensor = None,
        padding_side: str = "left",
        num_beams: int = 1,
    ):
        """
        Insert the vision tokens directly into the language stream/
        This requires us to modify the input_ids, attention_mask, and labels.
        """
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            assert attention_mask.shape[1] == past_len + lang_x.shape[1], (
                "Attention_mask must be as long as the entire past len (including image tokens) and current input IDs. "
                + "Check that you've expanded the attention mask to account for past image tokens."
            )

        if vision_tokens is None:
            return {
                "input_ids": lang_x,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        # get the language embeddings
        lang_embeds = self.lang_model.get_input_embeddings()(lang_x)

        # build up the multimodal embeddings
        B = lang_x.shape[0]
        has_labels = labels is not None
        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = [] if has_labels else None
        for i in range(B):
            # get index of <image> tokens in lang_x[i]
            image_token_idxs = torch.where(lang_x[i] == self.media_token_id)[0]

            if len(image_token_idxs) == 0:
                multimodal_embeds.append(lang_embeds[i].clone())
                multimodal_attention_mask.append(attention_mask[i].clone())
                if has_labels:
                    multimodal_labels.append(labels[i].clone())
                continue

            # since an image is represented by self.num_tokens_per_vis tokens, we need to offset the image_token_idxs
            for j, img_idx in enumerate(image_token_idxs):
                image_token_idxs[j] += (
                    self.num_tokens_per_vis - 1
                ) * j  # FIXME: different offset for any resolution encoding when has multiple images.

            # loop through the image_token_idxs and insert the vision tokens
            new_embed = lang_embeds[i].clone()
            new_attention_mask = (
                attention_mask[i].clone() if attention_mask is not None else None
            )
            if has_labels:
                new_label = labels[i].clone()

            for img_num, img_idx in enumerate(image_token_idxs):
                if img_num > 0:
                    # FIXME: hardcoded as such to avoid assertion error, but this only works for single image samples.
                    break
                # Get vision token attention mask for padded llava-style any resolution image tokens.
                if self.image_aspect_ratio == "anyres":
                    num_vis_tokens = vision_tokens[i][img_num].shape[0]
                    if vision_attention_mask is not None:
                        vis_attention_mask = vision_attention_mask[i]
                    else:
                        vis_attention_mask = torch.ones(
                            num_vis_tokens, dtype=torch.long
                        ).to(attention_mask.device)
                else:
                    assert (
                        vision_tokens[i][img_num].shape[0] == self.num_tokens_per_vis
                    ), f"vision token number mismatch: image embedding ({vision_tokens[i][img_num].shape[0]}) \
                            vs. model.num_tokens_per_vis ({self.num_tokens_per_vis})"
                    # By default, vision tokens are not padded.
                    num_vis_tokens = self.num_tokens_per_vis
                    vis_attention_mask = torch.ones(
                        num_vis_tokens, dtype=torch.long
                    ).to(attention_mask.device)

                new_embed = torch.cat(
                    (
                        new_embed[:img_idx],
                        vision_tokens[i][img_num],
                        new_embed[img_idx + 1 :],
                    ),
                    dim=0,
                )
                new_attention_mask = torch.cat(
                    (
                        new_attention_mask[:img_idx],
                        vis_attention_mask,
                        new_attention_mask[img_idx + 1 :],
                    ),
                    dim=0,
                )
                if has_labels:
                    new_label = torch.cat(
                        (
                            new_label[:img_idx],
                            torch.ones(num_vis_tokens, dtype=torch.long).to(
                                labels.device
                            )
                            * -100,
                            new_label[img_idx + 1 :],
                        ),
                        dim=0,
                    )
            multimodal_embeds.append(new_embed)
            multimodal_attention_mask.append(new_attention_mask)
            if has_labels:
                multimodal_labels.append(new_label)

        # stack
        multimodal_embeds = stack_with_padding(
            multimodal_embeds,
            padding_value=self.pad_token_id,
            padding_side=padding_side,
        )
        multimodal_attention_mask = stack_with_padding(
            multimodal_attention_mask,
            padding_value=0,
            padding_side=padding_side,
        )
        if has_labels:
            multimodal_labels = stack_with_padding(
                multimodal_labels,
                padding_value=-100,
                padding_side=padding_side,
            )

        return {
            "inputs_embeds": multimodal_embeds,
            "attention_mask": multimodal_attention_mask,
            "labels": multimodal_labels,
        }

    def _postprocess_outputs_from_forward(
        self,
        output: CausalLMOutputWithPast,
        lang_x: torch.Tensor,
        vision_tokens: torch.Tensor,
        past_vision_tokens: torch.Tensor,
        past_media_locations: torch.Tensor,
        use_cache: bool = False,
    ):
        # Include the past vision tokens and past media locations in the output
        updated_vision_tokens, updated_media_locations = self._concat_vision_cache(
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
            use_cache=use_cache,
        )

        # return logits that are the same shape as the original input_ids
        logits = output.logits
        batch_logits = []
        B, T_txt = lang_x.shape
        for i in range(B):
            sequence_logits = []
            logits_j = 0
            for j in range(T_txt):
                if lang_x[i, j] != self.media_token_id:
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += 1
                else:
                    # append the logit for the first image token, then skip over the rest
                    # note: the model actually learns to predict <im_patch>, not <image>
                    sequence_logits.append(logits[i, logits_j])
                    logits_j += self.num_tokens_per_vis
            sequence_logits = torch.stack(sequence_logits, dim=0)  # (B, vocab_size)
            batch_logits.append(sequence_logits)

        batch_logits = torch.stack(batch_logits, dim=0)  # (B, T_txt, vocab_size)
        # The final logits shape should be the same as the original input_ids shape
        assert batch_logits.shape[:2] == (B, T_txt)

        # assemble the output
        output = VLMOutputWithPast(
            loss=output.loss,
            logits=batch_logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            past_media_locations=updated_media_locations,
            past_vision_tokens=updated_vision_tokens,
        )

        return output

    def _post_forward_hook(self):
        pass

    @property
    def num_params_per_module(self):
        """Print the number of parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder):,} parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer):,} parameters",
                f"Language model: {num_params(self.lang_model):,} parameters",
            ]
        )

    @property
    def num_trainable_params_per_module(self):
        """Print the number of trainable parameters per module in the model"""
        return "\n".join(
            [
                f"Vision encoder: {num_params(self.vision_encoder, filter_to_trainable=True):,} trainable parameters",
                f"Vision tokenizer: {num_params(self.vision_tokenizer, filter_to_trainable=True):,} trainable parameters",
                f"Language model: {num_params(self.lang_model, filter_to_trainable=True):,} trainable parameters",
            ]
        )


class KosmosInstruct(VLMWithLanguageStream):
    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_tokenizer: nn.Module,
        lang_model: nn.Module,
        initial_tokenizer_len: int,
        pad_token_id: int,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        image_aspect_ratio: str = "pad",
        anyres_patch_sampling: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            padding_token_id (int): id of the padding token. None if no padding token; then a padding token
                will be inserted into self.special_tokens, which factory.py fills after creating new tokens
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "media_token": "<image>",
            "image_placeholder_token": "<image placeholder>",
            "end_of_trunk_token": "<|endofchunk|>",
        }
        lang_embedding_dim = lang_model.get_input_embeddings().weight.shape[1]
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=vision_tokenizer,
            lang_model=lang_model,
            initial_tokenizer_len=initial_tokenizer_len,
            gradient_checkpointing=gradient_checkpointing,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        self.image_aspect_ratio = image_aspect_ratio
        self.anyres_patch_sampling = anyres_patch_sampling

    def set_trainable(self):
        """
        Unfreeze everything except the vision_encoder
        """
        self.requires_grad_(True)
        self.vision_encoder.requires_grad_(False)

    def _should_apply_weight_decay(self, parameter_name):
        """
        Kosmos applies 0.01 weight deacy to everything
        """
        return True

    def forward(
        self,
        vision_x: Optional[torch.Tensor],
        lang_x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple] = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Args:
            vision_x: Vision input
                shape (B, T_img, F, C, H, W) with F=1
                only F = 1 is supported (single-frame videos)
                if T_img > the number of media tokens in the corresponding input_ids (lang_x),
                only the first number of media tokens in lang_x are used
            lang_x: Language input ids, with media tokens denoting where
                visual media should be inserted.
                shape (B, T_txt)
            attention_mask: Attention mask. Defaults to None.
            labels: Labels. Defaults to None.
                shape (B, T_txt)
            past_key_values (Tuple[torch.Tensor]], optional): Past key value pairs for each of the T_txt previous tokens in the language model. Defaults to None.
                list of length = number of decoder layers in the LM
                exact implementation depends on LM, see Hugging Face docs
            past_media_locations (torch.Tensor, optional): boolean mask denoting which of the previous T_txt tokens were media tokens. Defaults to None.
                shape (B, T_txt)
            past_vision_tokens (torch.Tensor, optional): Previous vision tokens. Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to False.
                If True, includes key_values, media_locations, and vision_tokens in the output.
        """
        assert not (past_vision_tokens is None) ^ (
            past_media_locations is None
        ), "past_vision_tokens and past_media_locations must both be None or both be not None"

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == "anyres":
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(
                    input_dict, lang_x.device
                )
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[
                    :, None, None, :, :
                ]  # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            # Prepare text embeds for instruction-aware image query sampling.
            # FIXME: for debugging purposed, truncating text input to vision tokenizer to be 256 at max.
            lang_x_truncated = lang_x[:, :256]
            text_embeds = self.lang_model.get_input_embeddings()(lang_x_truncated)
            # TODO: repeat text_embeds to match the number of patches for each image patch group.
            if self.anyres_patch_sampling:
                repeated_text_embeds = []
                for i, np in enumerate(split_sizes):
                    repeated_text_embeds.append(text_embeds[i].repeat(np, 1, 1))
                text_embeds = torch.cat(repeated_text_embeds, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, text_embeds)

            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                max_n_vis_token = max(
                    [vis.shape[0] * vis.shape[-2] for vis in vision_token_groups]
                )
                # Padding.
                padded_vision_tokens = []
                padded_attn_masks = []
                for patch_vis_tokens in vision_token_groups:
                    patch_vis_tokens = patch_vis_tokens.flatten(
                        0, 2
                    )  # [Np, 1, v, d] -> [Np*v, d]
                    n_vis_token = patch_vis_tokens.shape[0]
                    patch_attn = torch.ones(
                        (max_n_vis_token),
                        dtype=torch.long,
                        device=patch_vis_tokens.device,
                    )
                    if n_vis_token < max_n_vis_token:
                        padded_vis_token = torch.zeros(
                            (max_n_vis_token, patch_vis_tokens.shape[-1]),
                            dtype=patch_vis_tokens.dtype,
                            device=patch_vis_tokens.device,
                        )
                        padded_vis_token[:n_vis_token, :] = patch_vis_tokens
                        patch_attn[n_vis_token:] = 0
                    else:
                        padded_vis_token = patch_vis_tokens
                    padded_vision_tokens.append(padded_vis_token)
                    padded_attn_masks.append(patch_attn)
                vision_tokens = torch.stack(padded_vision_tokens, dim=0)
                vision_attention_mask = torch.stack(padded_attn_masks, dim=0)
                vision_tokens = vision_tokens[:, None, :, :]
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            padding_side="right",
            past_vision_tokens=past_vision_tokens,
        )
        output = self.lang_model(
            **new_inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )

        # postprocessing may be needed, e.g. to remove extra tokens from logits that were inserted into the language stream
        # or to add the past_vision_tokens and past_media_locations to the output
        output = self._postprocess_outputs_from_forward(
            output=output,
            lang_x=lang_x,
            vision_tokens=vision_tokens,
            use_cache=use_cache,
            past_vision_tokens=past_vision_tokens,
            past_media_locations=past_media_locations,
        )

        # postforward hooks
        self._post_forward_hook()
        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        image_size: Optional[Tuple] = None,
        attention_mask: torch.Tensor = None,
        past_key_values: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor]]]
        ] = None,
        past_media_locations: Optional[torch.Tensor] = None,
        past_vision_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                see documentation for forward
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            **kwargs: see generate documentation in Hugging Face CausalLM models.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)

        # convert pixels to vision tokens
        vision_attention_mask = None
        if vision_x is not None:
            if self.image_aspect_ratio == "anyres":
                input_dict = dict(image=vision_x, image_size=image_size)
                vision_features, vision_attn_masks = self._encode_vision_x_anyres(
                    input_dict, lang_x.device
                )
            else:
                vision_features = self._encode_vision_x(vision_x=vision_x)
                vision_attn_masks = None
            if self.anyres_patch_sampling:
                split_sizes = [feature.shape[0] for feature in vision_features]
                vision_features = torch.cat(vision_features, dim=0)
                vision_features = vision_features[
                    :, None, None, :, :
                ]  # Expand dimensions.
                vision_attn_masks = torch.cat(vision_attn_masks, dim=0)
            # Prepare text embeds for instruction-aware image query sampling.
            lang_x_truncated = lang_x[:, :256]
            text_embeds = self.lang_model.get_input_embeddings()(
                lang_x_truncated
            )  # FIXME: check function calling.
            # Repeat text_embeds to match the number of patches for each image patch group.
            if self.anyres_patch_sampling:
                repeated_text_embeds = []
                for i, np in enumerate(split_sizes):
                    repeated_text_embeds.append(text_embeds[i].repeat(np, 1, 1))
                text_embeds = torch.cat(repeated_text_embeds, dim=0)
            vision_tokens = self.vision_tokenizer(vision_features, text_embeds)

            # Post-processing: Split the batches into groups of patches and concatenate them together.
            if self.anyres_patch_sampling:
                vision_token_groups = torch.split(vision_tokens, split_sizes, dim=0)
                max_n_vis_token = max(
                    [vis.shape[0] * vis.shape[-2] for vis in vision_token_groups]
                )
                # Padding.
                padded_vision_tokens = []
                padded_attn_masks = []
                for patch_vis_tokens in vision_token_groups:
                    patch_vis_tokens = patch_vis_tokens.flatten(
                        0, 2
                    )  # [Np, 1, v, d] -> [Np*v, d]
                    n_vis_token = patch_vis_tokens.shape[0]
                    patch_attn = torch.ones(
                        (max_n_vis_token),
                        dtype=torch.long,
                        device=patch_vis_tokens.device,
                    )
                    if n_vis_token < max_n_vis_token:
                        padded_vis_token = torch.zeros(
                            (max_n_vis_token, patch_vis_tokens.shape[-1]),
                            dtype=patch_vis_tokens.dtype,
                            device=patch_vis_tokens.device,
                        )
                        padded_vis_token[:n_vis_token, :] = patch_vis_tokens
                        patch_attn[n_vis_token:] = 0
                    else:
                        padded_vis_token = patch_vis_tokens
                    padded_vision_tokens.append(padded_vis_token)
                    padded_attn_masks.append(patch_attn)
                vision_tokens = torch.stack(padded_vision_tokens, dim=0)
                vision_attention_mask = torch.stack(padded_attn_masks, dim=0)
                vision_tokens = vision_tokens[:, None, :, :]
        else:
            vision_tokens = None

        # fuse the vision and language tokens
        # for xattn, vision_x and media_location are repeat_interleaved s.t.
        # the total batch size is B * num_beams
        new_inputs = self._prepare_inputs_for_forward(
            vision_tokens=vision_tokens,
            lang_x=lang_x,
            attention_mask=attention_mask,
            vision_attention_mask=vision_attention_mask,
            past_key_values=past_key_values,
            past_media_locations=past_media_locations,
            past_vision_tokens=past_vision_tokens,
            padding_side="left",
            num_beams=num_beams,
        )
        if transformers.__version__ == "4.41.0.dev0":
            output = self.lang_model.generate(
                **new_inputs,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        else:
            output = self.lang_model.generate(
                **new_inputs,
                past_key_values=past_key_values,
                num_beams=num_beams,
                use_cache=True,
                **kwargs,
            )
        self._post_forward_hook()
        return output
