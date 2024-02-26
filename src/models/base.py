import abc
import torch.nn


class VisionLanguageModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute_loss(self, images, prompts, targets):
        pass

    @abc.abstractmethod
    def generate(self, images, prompts):
        pass
