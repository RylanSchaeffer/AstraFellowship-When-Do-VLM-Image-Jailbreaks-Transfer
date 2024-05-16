import torch
import ast
import math
from PIL import Image


def has_fn(model, fn_name):
    """Check if model has a function fn_name"""
    return callable(getattr(model, fn_name, None))


def exists(val):
    return val is not None


def num_params(module, filter_to_trainable=False):
    """Returns the number of parameters in the module, or optionally only the trainable parameters"""
    if filter_to_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def hasattr_recursive(obj, att):
    """
    Check if obj has nested attribute
    Example: hasattr_recursive(obj, 'a.b.c') is equivalent to hasattr(obj, 'a') and hasattr(obj.a, 'b') and hasattr(obj.a.b, 'c')
    """
    if att == "":
        return True
    i = att.find(".")
    if i < 0:
        return hasattr(obj, att)
    else:
        try:
            return hasattr_recursive(getattr(obj, att[:i]), att[i + 1 :])
        except:
            return False


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def stack_with_padding(list_of_tensors, padding_value=0, padding_side="right"):
    """
    Stack a list of tensors with padding on one side
    Args:
        list_of_tensors (list[torch.Tensor]): List of tensors to stack
        padding_value (int, optional): Value to pad with. Defaults to 0.
        padding_side (str, optional): Side to pad on. Defaults to "right".
    Returns:
        torch.Tensor: Stacked tensors
    """
    max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
    padded_tensors = []
    for tensor in list_of_tensors:
        num_tokens = tensor.size(0)
        if len(tensor.size()) == 1:
            padding = torch.full(
                (max_tokens - num_tokens,),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            padding = torch.full(
                (max_tokens - num_tokens, tensor.size(1)),
                padding_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        padded_tensor = (
            torch.cat((tensor, padding), dim=0)
            if padding_side == "right"
            else torch.cat((padding, tensor), dim=0)
        )
        padded_tensors.append(padded_tensor)
    return torch.stack(padded_tensors)


def check_embedding_fns(lang_model):
    """Checks for and attempts to set {get/set}_{input/output}_embeddings functions to the model"""
    if not has_fn(lang_model, "get_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.get_input_embeddings = lambda: lang_model.transformer.wte
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.get_input_embeddings = lambda: lang_model.decoder.embed_tokens
        else:
            raise ValueError(
                "We require the language encoder to have a get_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_input_embeddings"):
        if hasattr_recursive(lang_model, "transformer.wte"):  # MPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "transformer.wte", x
            )
        elif hasattr_recursive(lang_model, "model.decoder.embed_tokens"):  # OPT
            lang_model.set_input_embeddings = lambda x: setattr_recursive(
                lang_model, "model.decoder.embed_tokens", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_input_embeddings method but we couldn't determine the name of the input embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "get_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.get_output_embeddings = lambda: lang_model.lm_head
        else:
            raise ValueError(
                "We require the language encoder to have a get_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )

    if not has_fn(lang_model, "set_output_embeddings"):
        if hasattr_recursive(lang_model, "lm_head"):
            lang_model.set_output_embeddings = lambda x: setattr_recursive(
                lang_model, "lm_head", x
            )
        else:
            raise ValueError(
                "We require the language encoder to have a set_output_embeddings method but we couldn't determine the name of the output embeddings attribute. Please supply this manually in factory.py."
            )


# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


def unpad_image(tensor, original_size, keep_original_shape=False):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        if keep_original_shape:
            attention_mask = torch.ones(
                (current_height, current_width), device=tensor.device
            )
            attention_mask[:padding, :] = 0
            attention_mask[current_height - padding :, :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
            return unpadded_tensor, None
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        if keep_original_shape:
            attention_mask = torch.ones(
                (current_height, current_width), device=tensor.device
            )
            attention_mask[:, :padding] = 0
            attention_mask[:, current_width - padding :] = 0
            return tensor, attention_mask
        else:
            unpadded_tensor = tensor[:, :, padding : current_width - padding]
            return unpadded_tensor, None


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # FIXME: determine grid_pinpoints from image sizes.
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    processor_size = processor.transforms[0].size
    patches = divide_to_patches(image_padded, processor_size[0])

    image_original_resize = image.resize((processor_size[0], processor_size[0]))

    image_patches = [image_original_resize] + patches
    image_patches = [processor(image_patch) for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.transforms[-1].mean)
            )
            image = image_processor(image)
            new_images.append(image)
    elif image_aspect_ratio in ["anyres", "anyres-legacy"]:
        base_img_size = image_processor.transforms[0].size[0]
        for image in images:
            image = process_anyres_image(
                image,
                image_processor,
                [
                    [base_img_size, base_img_size * 2],
                    [base_img_size * 2, base_img_size],
                    [base_img_size * 2, base_img_size * 2],
                    [base_img_size * 3, base_img_size],
                    [base_img_size, base_img_size * 3],
                ],
            )

            # Debug any res inference by only using 672x672.
            # image = process_anyres_image(image, image_processor, [[base_img_size*2,base_img_size*2]])
            new_images.append(image)
    else:
        return image_processor(images)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


# define the prompt template
def apply_xgen_prompt_template(prompt: str):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


def apply_xgen_prompt_template_with_target(prompt: str, target: str):
    # Space infront of target
    print(f"Got target: {target}")
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n{target}"
    )
    return s
