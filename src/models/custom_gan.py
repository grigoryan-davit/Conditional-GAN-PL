from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn


class UnetBlock(nn.module):
    def __init__(
        self,
        num_filters: int,
        num_inputs: int,
        submodule=None,
        input_channels: Optional[int] = None,
        dropout: bool = False,
        innermost: bool = False,
        outermost: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_channels is None:
            input_channels = num_filters
        downconv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_inputs,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        downrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        downnorm = nn.BatchNorm2d(num_features=num_inputs)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(num_features=num_filters)

        if outermost:
            upconv = nn.ConvTranspose2d(
                in_channels=num_inputs * 2,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            model = [downconv] + [submodule] + [uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(
                in_channels=num_inputs,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(
                in_channels=num_inputs * 2,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = [downrelu, downconv, downnorm] + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    def __init__(
        self,
        input_c: int = 1,
        output_c: int = 2,
        n_down: int = 8,
        num_filters: int = 64,
    ):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True
            )
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(
            num_filters=output_c,
            num_inputs=out_filters,
            input_channels=input_c,
            submodule=unet_block,
            outermost=True,
        )

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int, num_filters: int = 64, n_down: int = 3):
        super().__init__()
        model = [
            self.get_layers(
                self, num_inputs=input_channels, num_filters=num_filters, norm=False
            )
        ]
        model += [
            self.get_layers(
                num_filters * 2**i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [
            self.get_layers(num_filters * 2**n_down, 1, s=1, norm=False, act=False)
        ]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(
        self,
        num_inputs: int,
        num_filters: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        norm: bool = True,
        act: bool = True,
    ) -> nn.Sequential:  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(
                in_channels=num_inputs,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                paddding=padding,
                bias=not norm,
            )
        ]  # it's always helpful to make a separate method for that purpose
        if norm:
            layers += [nn.BatchNorm2d(num_filters)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
