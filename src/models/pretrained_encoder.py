from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import transforms


# Размерности CLS-эмбеддинга для поддерживаемых checkpoints DINOv2.
_SUPPORTED_ENCODERS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_PATCH_SIZE = 14
DEFAULT_DINOV2_IMAGE_SIZE = 224


@dataclass(frozen=True)
class PretrainedEncoderInfo:
    name: str
    embedding_dim: int
    patch_size: int = DINOV2_PATCH_SIZE
    recommended_image_size: int = (
        DEFAULT_DINOV2_IMAGE_SIZE
    )


class FrozenDinoV2Encoder(nn.Module):
    """
    Замороженный self-supervised encoder DINOv2 (Oquab et al., 2023).

    В отличие от ConvolutionalVAE в этом проекте, DINOv2 не обучается
    на CRC-датасете: веса загружаются из общедоступного checkpoint,
    предобученного на естественных изображениях (не на гистологии),
    и используются только для inference ("frozen backbone probing").

    Мотивация: в описании вакансии single-modality encoders явно
    перечислены как VAE, LIAE и DINOv2. Этот класс добавляет второй
    тип encoder, чтобы можно было честно сравнить:

        обучаемое, но узкоспециализированное представление (VAE)

    против

        репрезентацию большой предобученной self-supervised модели
        без дообучения на гистологии.

    DINOv2 не обучался на H&E-изображениях, поэтому перенос на
    гистологические patches не гарантирован — сравнение этого
    baseline с VAE и с color-шорткатом (RGB-HSV PCA) само по себе
    является содержательным экспериментом, а не просто заменой VAE
    на "более сильную" модель.

    Требует:
        - пакет torch>=2.0 и torchvision;
        - интернет-доступ при первом вызове (torch.hub скачивает
          веса из github.com/facebookresearch/dinov2).
    """

    def __init__(
        self,
        encoder_name: str = "dinov2_vits14",
    ) -> None:
        super().__init__()

        if encoder_name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                "Unsupported encoder_name. "
                f"Expected one of "
                f"{sorted(_SUPPORTED_ENCODERS)}, "
                f"received {encoder_name!r}."
            )

        self.encoder_name = encoder_name
        self.embedding_dim = _SUPPORTED_ENCODERS[
            encoder_name
        ]

        try:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2",
                encoder_name,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load DINOv2 weights via "
                "torch.hub. This requires internet "
                "access on first use "
                "(github.com/facebookresearch/dinov2). "
                f"Original error: {error}"
            ) from error

        self.backbone.eval()

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs:
            Tensor формы batch_size x 3 x H x W,
            ImageNet-нормализованный. H и W должны быть кратны
            patch_size (14). Используйте
            build_dinov2_preprocessing_transform для получения
            корректного препроцессинга.

        Returns
        -------
        Tensor формы batch_size x embedding_dim (CLS-токен).
        """
        if self.training:
            raise RuntimeError(
                "FrozenDinoV2Encoder must stay in eval "
                "mode. Call .eval() before use."
            )

        return self.backbone(inputs)


def build_pretrained_encoder(
    encoder_name: str,
    device: torch.device,
) -> FrozenDinoV2Encoder:
    """
    Создаёт замороженный encoder и перемещает его на нужное устройство.
    """
    encoder = FrozenDinoV2Encoder(
        encoder_name=encoder_name
    )

    encoder.to(device)
    encoder.eval()

    return encoder


def get_pretrained_encoder_info(
    encoder_name: str,
) -> PretrainedEncoderInfo:
    if encoder_name not in _SUPPORTED_ENCODERS:
        raise ValueError(
            "Unsupported encoder_name. "
            f"Expected one of "
            f"{sorted(_SUPPORTED_ENCODERS)}, "
            f"received {encoder_name!r}."
        )

    return PretrainedEncoderInfo(
        name=encoder_name,
        embedding_dim=_SUPPORTED_ENCODERS[
            encoder_name
        ],
    )


def build_dinov2_preprocessing_transform(
    image_size: int = DEFAULT_DINOV2_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Строит deterministic-препроцессинг для DINOv2.

    В отличие от evaluation-препроцессинга VAE в этом проекте
    (build_evaluation_transforms), DINOv2 ожидает изображения,
    нормализованные ImageNet-статистикой, а не [0, 1] tensors.
    """
    if image_size <= 0:
        raise ValueError(
            "image_size must be positive."
        )

    if image_size % DINOV2_PATCH_SIZE != 0:
        raise ValueError(
            "image_size must be divisible by the "
            f"DINOv2 patch size ({DINOV2_PATCH_SIZE}). "
            f"Received {image_size}."
        )

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )
