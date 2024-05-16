import random
from typing import List, Optional, Sequence, Tuple, Union
import torchvision.transforms.functional as F
from torchvision.transforms import (
    Normalize,
    Compose,
    InterpolationMode,
    ToTensor,
    Resize,
    CenterCrop,
)
import numbers
import torch
import ast
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.utils import TensorType
from torchvision import transforms


class Blip3ImageProcessor(BaseImageProcessor):

    def __init__(
        self,
        do_resize: bool = True,
        resize_mode: str = "squash",
        interpolation_mode: str = "bicubic",
        size: Union[Tuple[int, int], List[int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resize_mode = resize_mode
        self.interpolation_mode = interpolation_mode
        self.size = size if size is not None else (378, 378)
        self.image_mean = (
            image_mean
            if image_mean is not None
            else [0.48145466, 0.4578275, 0.40821073]
        )
        self.image_std = (
            image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]
        )

    @classmethod
    def resize(cls, image_size, resize_mode, interpolation="bicubic", fill_color=0):
        interpolation_mode = (
            InterpolationMode.BILINEAR
            if interpolation == "bilinear"
            else InterpolationMode.BICUBIC
        )
        if resize_mode == "longest":
            transforms = [
                ResizeKeepRatio(
                    image_size, interpolation=interpolation_mode, longest=1
                ),
                CenterCropOrPad(image_size, fill=fill_color),
            ]
        elif resize_mode == "squash":
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        else:
            assert resize_mode == "shortest"
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transforms = [Resize(image_size[0], interpolation=interpolation_mode)]
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms = [ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]
        return transforms

    # @classmethod
    # def convert_rgb(cls, image):
    #     return image.convert("RGB")

    def _preprocess(self, images: ImageInput) -> torch.Tensor:
        """
        Preprocesses the input images by resizing, converting to RGB, and normalizing.

        Args:
            images (ImageInput): The input images to be preprocessed.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        # Define the list of transformations
        transforms_list = self.resize(
            self.size, self.resize_mode, self.interpolation_mode
        )
        transforms_list.extend([Normalize(mean=self.image_mean, std=self.image_std)])

        # Compose the transformations
        composed_transforms = Compose(transforms_list)

        # Apply the composed transformations to the images
        images_tensor = composed_transforms(images)

        return images_tensor

    def preprocess(
        self,
        images: torch.Tensor,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        assert (
            images.dim() == 4
        ), f"Input images should have 4 dimensions, got {images.dim()}"
        if "image_aspect_ratio" in kwargs:
            image_aspect_ratio = kwargs["image_aspect_ratio"]
        else:
            image_aspect_ratio = "pad"
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self._preprocess(image)
                new_images.append(image)
        else:
            if isinstance(self.size, (tuple, list)):
                base_img_size = self.size[0]
            else:
                raise ValueError("size should be list or tuple")
            for image in images:
                image = process_anyres_image(
                    image,
                    self._preprocess,
                    self.size,
                    [
                        [base_img_size, base_img_size * 2],
                        [base_img_size * 2, base_img_size],
                        [base_img_size * 2, base_img_size * 2],
                        [base_img_size * 3, base_img_size],
                        [base_img_size, base_img_size * 3],
                    ],
                )
                new_images.append(image)

        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        if image_aspect_ratio == "pad":
            new_images = BatchFeature(
                data={"pixel_values": new_images.unsqueeze(0).unsqueeze(0)},
                tensor_type=return_tensors,
            )
        else:
            new_images = BatchFeature(
                data={"pixel_values": new_images.unsqueeze(0)},
                tensor_type=return_tensors,
            )
        return new_images


class ResizeKeepRatio:
    """Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation=InterpolationMode.BICUBIC,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
        img,
        target_size,
        longest,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        """Get parameters"""
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1.0, 1.0)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(
                random_aspect_range[0], random_aspect_range[1]
            )
            ratio_factor = (
                ratio_factor[0] / aspect_factor,
                ratio_factor[1] * aspect_factor,
            )
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (tensor): Image to be cropped and resized.

        Returns:
            tensor: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img,
            self.size,
            self.longest,
            self.random_scale_prob,
            self.random_scale_range,
            self.random_aspect_prob,
            self.random_aspect_range,
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += f", interpolation={self.interpolation})"
        format_string += f", longest={self.longest:.3f})"
        return format_string


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def center_crop_or_pad(
    img: torch.Tensor, output_size: List[int], fill=0
) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def process_anyres_image(image_tensor, processor, processor_size, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        processor: The image processor object.
        processor_size (tuple, list): The size of the image processor.
        grid_pinpoints (str or list): A string representation or list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Determine possible resolutions from grid_pinpoints
    if isinstance(grid_pinpoints, list):
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)

    best_resolution = select_best_resolution(
        image_tensor.shape[1:], possible_resolutions
    )
    image_padded = resize_and_pad_image(image_tensor, best_resolution)

    # Divide the padded image into patches
    patches = divide_to_patches(image_padded, processor_size[0])

    # Resize the original image to processor size
    resize_transform = transforms.Resize((processor_size[0], processor_size[0]))
    image_original_resize = resize_transform(image_tensor)

    # Prepare patches for processing
    image_patches = [image_original_resize] + patches
    image_patches = [processor(image_patch) for image_patch in image_patches]

    return torch.stack(image_patches, dim=0)


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


def resize_and_pad_image(image_tensor, target_resolution):
    """
    Resize and pad an image tensor to a target resolution while maintaining aspect ratio.

    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        torch.Tensor: The resized and padded image tensor.
    """
    _, original_height, original_width = image_tensor.shape
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = int(original_height * scale_w)
    else:
        new_height = target_height
        new_width = int(original_width * scale_h)

    # Resize the image
    resize_transform = transforms.Resize((new_height, new_width))
    resized_image_tensor = resize_transform(image_tensor)

    # Pad the image
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    padding = (
        pad_x,
        pad_y,
        target_width - new_width - pad_x,
        target_height - new_height - pad_y,
    )
    padded_image_tensor = F.pad(resized_image_tensor, padding, fill=0)

    return padded_image_tensor


def divide_to_patches(image_tensor, patch_size):
    """
    Divides an image tensor into patches of a specified size.
    Args:
        image_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        patch_size (int): The size of each patch.
    Returns:
        list: A list of torch.Tensor objects representing the patches.
    """
    patches = []
    _, height, width = image_tensor.shape
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = image_tensor[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    return patches
