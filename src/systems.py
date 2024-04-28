import numpy as np
import lightning
import torch
import torch.optim
import torchvision.transforms
from typing import Any, Dict, List, Mapping, Optional, Tuple
import wandb

from src.models.ensemble import VLMEnsemble
from src.utils import create_initial_image


class VLMEnsembleAttackingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        model_device: Mapping[str, torch.device]  = {},
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["models_to_attack"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
            precision=wandb_config["lightning_kwargs"]["precision"],
            model_device=model_device,
        )
        # Load initial image plus prompt and target data.
        tensor_image: torch.Tensor = create_initial_image(
            image_kwargs=wandb_config["image_kwargs"],
            seed=wandb_config["seed"],
        )
        self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=True)
        self.convert_tensor_to_pil_image = torchvision.transforms.ToPILImage()
        self.optimizer_step_counter = 0

    def configure_optimizers(self) -> Dict:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        # TODO: Maybe add SWA
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging
        optimization_kwargs = self.wandb_config["optimization"]
        if optimization_kwargs["optimizer"] == "adadelta":
            optimizer = torch.optim.Adadelta(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
            )
        elif optimization_kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                eps=optimization_kwargs[
                    "eps"
                ],  # https://stackoverflow.com/a/42420014/4570472
            )
        elif optimization_kwargs["optimizer"] == "rmsprop":
            optimizer = torch.optim.RMSprop(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
                eps=1e-4,
            )
        elif optimization_kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                [self.tensor_image],
                lr=optimization_kwargs["learning_rate"],
                weight_decay=optimization_kwargs["weight_decay"],
                momentum=optimization_kwargs["momentum"],
            )
        else:
            # TODO: add adafactor https://pytorch-optimizer.readthedocs.io/en/latest/index.html
            raise NotImplementedError(f"{self.wandb_config['optimizer']}")

        optimizer_and_maybe_others_dict = {
            "optimizer": optimizer,
        }

        # if self.wandb_config["learning_rate_scheduler"] is None:
        #     pass
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "cosine_annealing_warm_restarts"
        # ):
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer=optimizer,
        #         T_0=2,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif (
        #     self.wandb_config["learning_rate_scheduler"]
        #     == "linear_warmup_cosine_annealing"
        # ):
        #     from flash.core.optimizers import LinearWarmupCosineAnnealingLR
        #
        #     scheduler = LinearWarmupCosineAnnealingLR(
        #         optimizer=optimizer,
        #         warmup_epochs=1,
        #         max_epochs=self.wandb_config["n_epochs"],
        #     )
        #
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #
        # elif self.wandb_config["learning_rate_scheduler"] == "reduce_lr_on_plateau":
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         factor=0.95,
        #         optimizer=optimizer,
        #         patience=3,
        #     )
        #     optimizer_and_maybe_others_dict["lr_scheduler"] = scheduler
        #     optimizer_and_maybe_others_dict["monitor"] = "train/loss=total_loss"
        # else:
        #     raise NotImplementedError(f"{self.wandb_config['learning_rate_scheduler']}")

        return optimizer_and_maybe_others_dict

    def training_step(
        self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
        losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
            image=self.tensor_image,
            text_data_by_model=batch,
        )
        for loss_str, loss_val in losses_per_model.items():
            self.log(
                f"loss/{loss_str}",
                loss_val.detach().item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        self.log(
            "optimizer_step_counter",
            self.optimizer_step_counter,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # if batch_idx == 0:
        #     print(torch.cuda.memory_summary())
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

        return losses_per_model["avg"]

    def optimizer_step(self, *args, **kwargs):
        if (
            self.optimizer_step_counter
            % self.wandb_config["lightning_kwargs"]["log_image_every_n_steps"]
        ) == 0:
            wandb.log(
                {
                    f"jailbreak_image_step={self.optimizer_step_counter}": wandb.Image(
                        # https://docs.wandb.ai/ref/python/data-types/image
                        # 0 removes the size-1 batch dimension.
                        # The transformation doesn't accept bfloat16.
                        data_or_path=self.convert_tensor_to_pil_image(
                            self.tensor_image[0].detach().to(torch.float32)
                        ),
                        # caption="Adversarial Image",
                    ),
                },
            )
        super().optimizer_step(*args, **kwargs)
        self.optimizer_step_counter += 1
        with torch.no_grad():
            self.tensor_image.data = self.tensor_image.data.clamp(min=0.0, max=1.0)


class VLMEnsembleEvaluatingSystem(lightning.LightningModule):
    def __init__(
        self,
        wandb_config: Dict[str, Any],
        tensor_image: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.wandb_config = wandb_config
        self.vlm_ensemble = VLMEnsemble(
            model_strs=wandb_config["model_to_eval"],
            model_generation_kwargs=wandb_config["model_generation_kwargs"],
        )
        # harmbench_evaluator = HarmBenchEvaluator()
        # llamaguard_evalutor = LlamaGuardEvaluator()
        self.tensor_image = torch.nn.Parameter(tensor_image, requires_grad=False)
        self.wandb_additional_data = {}

    def test_step(self, batch: Dict[str, Dict[str, torch.Tensor]], batch_idx: int):
        if self.tensor_image is None:
            raise ValueError("Image must be provided!")

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#training_step
        losses_per_model: Dict[str, torch.Tensor] = self.vlm_ensemble.compute_loss(
            image=self.tensor_image,
            text_data_by_model=batch,
        )

        for loss_str, loss_val in losses_per_model.items():
            self.log(
                f"loss/{loss_str}",
                loss_val.detach().item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # Make sure the number of optimizer steps is simultaneously logged.
        self.log(
            "optimizer_step_counter",
            self.wandb_additional_data["optimizer_step_counter"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
