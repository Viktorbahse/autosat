"""U-Net inspired encoder-decoder architecture with a ResNet encoder as proposed by Alexander Buslaev.

See:
- https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/abs/1411.4038  - Fully Convolutional Networks for Semantic Segmentation
- https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
- https://arxiv.org/abs/1801.05746 - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/abs/1806.00844 - TernausNetV2: Fully Convolutional Network for Instance Segmentation

"""

import torch
from torchvision.models import ResNet50_Weights, resnet50


class ConvRelu(torch.nn.Module):
    """3x3 convolution followed by ReLU activation building block."""

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """
        super().__init__()
        self.block = torch.nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return torch.nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(torch.nn.Module):
    """Decoder building block upsampling resolution by a factor of two."""

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.
        Args: num_in: number of input feature maps, num_out: number of output feature maps
        """
        super().__init__()
        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return self.block(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(torch.nn.Module):
    """The "U-Net" architecture for semantic segmentation,
    adapted by changing the encoder to a ResNet feature extractor.

    Also known as AlbuNet due to its inventor Alexander Buslaev.
    """

    def __init__(self, num_classes, num_filters=32, pretrained=True):
        """Creates an `UNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = resnet50(weights=weights)

        # keep two public attributes only: resnet and final
        self.out_ch = num_filters * 8

        # group decoder blocks to reduce public attributes count
        self._center = DecoderBlock(2048, self.out_ch)  # noqa: WPS432
        self._dec_blocks = torch.nn.ModuleList(
            [
                DecoderBlock(2048 + self.out_ch, self.out_ch),  # noqa: WPS432
                DecoderBlock(1024 + self.out_ch, self.out_ch),  # noqa: WPS432
                DecoderBlock(512 + self.out_ch, num_filters * 2),  # noqa: WPS432
                DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2),  # noqa: WPS432
                DecoderBlock(num_filters * 2 * 2, num_filters),  # noqa: WPS432
            ]
        )
        self._conv_relus = torch.nn.ModuleList(
            [
                ConvRelu(num_filters, num_filters),  # dec5
            ]
        )

        self.final = torch.nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):  # noqa: WPS210
        """The networks forward pass for which autograd synthesizes the backwards pass."""
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, "image resolution has to be divisible by 32 for resnet"  # noqa: WPS432

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self._center(torch.nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self._dec_blocks[0](torch.cat([enc4, center], dim=1))
        dec1 = self._dec_blocks[1](torch.cat([enc3, dec0], dim=1))
        dec2 = self._dec_blocks[2](torch.cat([enc2, dec1], dim=1))
        dec3 = self._dec_blocks[3](torch.cat([enc1, dec2], dim=1))
        dec4 = self._dec_blocks[4](dec3)
        dec5 = self._conv_relus[0](dec4)

        return self.final(dec5)
