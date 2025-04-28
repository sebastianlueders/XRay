import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze & Excitation Block: Adds channel-wise attention so that the model can learn which channels actually matter for the prediction
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            1
        )  # Global avg. pooling layer to reduce (H, W) to (1, 1)
        self.fc = nn.Sequential(
            nn.Linear(
                in_channels, in_channels // reduction, bias=False
            ),  # Shrinks feature size to compress information
            nn.ReLU(inplace=True),  # ReLU layer on compressed info to add non-linearity
            nn.Linear(
                in_channels // reduction, in_channels, bias=False
            ),  # Scales compressed feature back up to full size
            nn.Sigmoid(),  # Applies sigmoid to the outputs for interpreting them as "attention weights"
        )

    def forward(self, x):
        b, c, h, w = (
            x.size()
        )  # Obtain the batch size and the number of channels from the input tensor
        y = self.avg_pool(x).view(
            b, c
        )  # Average every channel over all pixels and flatten the result into a (BxC) tensor
        y = self.fc(y).view(
            b, c, 1, 1
        )  # Passes the flattened tensor through the fc layer to obtain channel attention weights from (0, 1) & then reshapes output back to (B x C x H(1) x W(1))
        return x * y.expand_as(
            x
        )  # Multiplies each channel by the obtained attention weight for the returned value


# Residual Block: Learns small modifications to input instead of full transformations
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.fir_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1
        )  # Applies 3x3 conv filter with specified in/out channels, stride and padding
        self.fir_bn = nn.BatchNorm2d(
            out_ch
        )  # Normalize first conv output across the batch
        self.sec_conv = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1
        )  # Applies another conv layer identitical to the first but with a hardcoded single stride
        self.sec_bn = nn.BatchNorm2d(
            out_ch
        )  # Normalize second conv output across the batch

        self.sc = (
            nn.Sequential()
        )  # Initialize the shortcut connection to an empty sequence (for identity mapping)
        if (
            stride != 1 or in_ch != out_ch
        ):  # If input/output dimensions will differ, apply necessary operations:
            self.sc = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=stride, padding=0
                ),  # Change the number of channels
                nn.BatchNorm2d(out_ch),  # Normalize the result by batch
            )

    def forward(self, x):
        out = F.relu(
            self.fir_bn(self.fir_conv(x))
        )  # Apply first conv and bn, followed by ReLu activation
        out = self.sec_bn(
            self.sec_conv(out)
        )  # Apply second conv and bn to the previous output
        out += self.sc(x)  # Add the shortcut (identity mapping or 1x1 conv + BN)
        out = F.relu(out)  # Run output through activation to introduce non-linearity
        return out


# Residual Network: Enables deep networks to learn faster by letting layers focus on learning residuals (small changes) through shortcut connections
class SEResNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=7):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, padding=1
        )  # Transforms input channels into 64 feature maps via a conv layer
        self.bn1 = nn.BatchNorm2d(64)  # Normalizes conv1 output
        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )  # Applies second conv layer for a 128 feature map output
        self.bn2 = nn.BatchNorm2d(128)  # Normalizes conv2 output
        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )  # Conv layer with 256 feature map output
        self.bn3 = nn.BatchNorm2d(256)  # Normalize
        self.relu = nn.ReLU(inplace=True)  # Defines an inplace ReLu activation
        self.se = SEBlock(
            256
        )  # Recalibrates the 256-channel feature maps by obtaining channel activation weights

        # Residual blocks double the number of channels each round and reduce spatial size of the images by a factor of 2
        self.res_block1 = ResidualBlock(256, 512, stride=2)
        self.res_block2 = ResidualBlock(512, 1024, stride=2)
        self.res_block3 = ResidualBlock(1024, 2048, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Ensures a fixed sized output of (2048x1x1)

        # Downscales the feature size through 3 fully connected layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        self.dropout1 = nn.Dropout(0.2)  # Light dropout used after conv layers
        self.dropout2 = nn.Dropout(
            0.5
        )  # Heavier dropout used in fc layers (overfitting prevention)
        self.fc4 = nn.Linear(
            256, num_classes
        )  # Final fc maps 256 features to the specified number of classes (final prediction values)

    def forward(self, x):

        # First pass (reduces spatial size by half)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Second pass (reduces spatial size by half)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Third pass (reduces spatial size by half)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        # Pass through SE Block
        x = self.se(x)

        # Pass through 3 residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Pool to 1x1 feature maps and flatten from (Bx2048x1x1) to (Bx2048)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Pass through fc layers with a higher dropout rate
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc4(x)

        return x
