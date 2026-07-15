from __future__ import annotations

from torch import nn


def get_group_count(
    number_of_channels: int,
    maximum_number_of_groups: int = 8,
) -> int:
    """
    Выбирает максимально возможное число групп для GroupNorm,
    которое без остатка делит число каналов.

    Примеры
    --------
    32 channels -> 8 groups
    16 channels -> 8 groups
    10 channels -> 5 groups
    """
    if number_of_channels <= 0:
        raise ValueError(
            "number_of_channels must be greater than zero."
        )

    if maximum_number_of_groups <= 0:
        raise ValueError(
            "maximum_number_of_groups must be greater than zero."
        )

    maximum_candidate = min(
        number_of_channels,
        maximum_number_of_groups,
    )

    for number_of_groups in range(
        maximum_candidate,
        0,
        -1,
    ):
        if number_of_channels % number_of_groups == 0:
            return number_of_groups

    return 1


class DownsampleBlock(nn.Module):
    """
    Уменьшает пространственное разрешение изображения в два раза.

    H x W -> H/2 x W/2
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=get_group_count(output_channels),
                num_channels=output_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)


class UpsampleBlock(nn.Module):
    """
    Увеличивает пространственное разрешение feature map в два раза.

    H x W -> 2H x 2W
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=get_group_count(output_channels),
                num_channels=output_channels,
            ),
            nn.SiLU(),
        )

    def forward(self, inputs):
        return self.block(inputs)