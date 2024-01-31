import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        depth=2,
    ):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2  # Padding is set to keep the shape of the input
        self.depth = depth  # Number of convolutional layers in the block

        self.block = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.block.append(
                    nn.Conv1d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    )
                )
            else:
                self.block.append(
                    nn.Conv1d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                    )
                )
            self.block.append(nn.BatchNorm1d(self.out_channels))
            self.block.append(nn.ReLU())

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        depth=1,
        output_padding=1,
        transpose_padding=None,
    ):
        """
        Stride, padding all refer to the convolutional layers, not the transpose convolutional layer
        """
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_padding = (
            kernel_size // 2
        )  # Padding is set to keep the shape of the input
        self.depth = depth  # Number of convolutions in the block
        self.output_padding = output_padding
        if transpose_padding is None:
            self.transpose_padding = (
                kernel_size // 2
            )  # Keeps the output size the same as the input size
        else:
            self.transpose_padding = transpose_padding

        # Stride set to 2 to upsample the input by a factor of 2
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels // 2,
            kernel_size=self.kernel_size,
            stride=2,
            padding=self.transpose_padding,
            output_padding=self.output_padding,
        )

        self.block = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.block.append(
                    nn.Conv1d(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.conv_padding,
                    )
                )
            else:
                self.block.append(
                    nn.Conv1d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.conv_padding,
                    )
                )
            self.block.append(nn.BatchNorm1d(self.out_channels))
            self.block.append(nn.ReLU())

    def forward(self, x, skip):
        """
        Forward call of the decoder block
        x: input tensor
        skip: tensor from the encoder block
        """
        x = self.conv_transpose(x)
        x = torch.cat((x, skip), dim=1)
        for layer in self.block:
            x = layer(x)
        return x


class Noise2Noise1D(nn.Module):
    def __init__(
        self,
        kernel_size,
        depth,
        num_input_features,
        num_blocks=4,
    ):
        super(Noise2Noise1D, self).__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.depth = depth
        self.num_blocks = num_blocks

        self.output_channels = 16

        self.encoder = nn.ModuleList()
        for block in range(self.num_blocks):
            if block == 0:
                self.encoder.append(
                    EncoderBlock(
                        in_channels=1,
                        out_channels=self.output_channels,
                        kernel_size=self.kernel_size,
                        depth=self.depth,
                    )
                )
            else:
                self.encoder.append(
                    EncoderBlock(
                        in_channels=self.output_channels,
                        out_channels=self.output_channels * 2,
                        kernel_size=self.kernel_size,
                        depth=self.depth,
                    )
                )
                self.output_channels *= 2

        self.latent_conv = nn.Conv1d(
            in_channels=self.output_channels,
            out_channels=self.output_channels * 2,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
        )
        self.output_channels *= 2

        self.decoder = nn.ModuleList()
        for block in range(self.num_blocks):
            # Using transposed convolutional layers to upsample the input by a factor of 2 in each block
            # Padding is set to keep the shape of the input (depending on if the number of features at that level of compression is even or odd)
            if num_input_features // (2 ** (num_blocks - (block + 1))) % 2 == 0:
                self.decoder.append(
                    DecoderBlock(
                        in_channels=self.output_channels,
                        out_channels=self.output_channels // 2,
                        kernel_size=self.kernel_size,
                        depth=self.depth,
                    )
                )
            else:
                self.decoder.append(
                    DecoderBlock(
                        in_channels=self.output_channels,
                        out_channels=self.output_channels // 2,
                        kernel_size=self.kernel_size,
                        transpose_padding=self.kernel_size // 2 - 1,
                        depth=self.depth,
                        output_padding=0,
                    )
                )
            self.output_channels //= 2

        self.out_conv = nn.Conv1d(
            in_channels=self.output_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        skip_connections = []
        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool1d(x, 2)

        x = self.latent_conv(x)

        for i, block in enumerate(self.decoder):
            x = block(x, skip_connections[-i - 1])

        x = self.out_conv(x)
        return x
