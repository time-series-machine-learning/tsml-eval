import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_filters,
        bottleneck_size,
        kernel_size,
        use_bottleneck=True,
    ):
        """
        Inception Module with optional bottleneck layer.
        """
        super().__init__()
        self.use_bottleneck = use_bottleneck

        # Bottleneck layer
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            if use_bottleneck
            else None
        )
        bottleneck_out_channels = bottleneck_size if use_bottleneck else in_channels

        # Convolutional layers with different kernel sizes
        kernel_sizes = [kernel_size // (2**i) for i in range(3)]
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1dSame(
                        bottleneck_out_channels,
                        num_filters,
                        kernel_size=k,
                        stride=1,
                        dilation=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )

        # Max pooling followed by 1x1 convolution
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(bottleneck_out_channels, num_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

    def forward(self, x):
        # Apply bottleneck if enabled
        if self.use_bottleneck:
            x = self.bottleneck(x)
        # Apply all inception sub-blocks
        out = [conv(x) for conv in self.convs]
        out.append(self.maxpool_conv(x))
        return torch.cat(out, dim=1)  # Concatenate along the channel dimension


class InceptionModel(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        depth=6,
        num_filters=32,
        use_residual=True,
        use_bottleneck=True,
        bottleneck_size=32,
        kernel_size=41,
    ):
        """
        Inception Model for 1D Time Series Classification.

        Parameters
        ----------
        num_channels : int
            Number of input channels in the time-series data.
        num_classes : int
            Number of output classes for classification.
        depth : int, default=6
            Number of inception blocks in the model.
        num_filters : int, default=32
            Number of filters in each convolutional layer.
        use_residual : bool, default=True
            Whether to include residual connections.
        use_bottleneck : bool, default=True
            Whether to include bottleneck layers in inception modules.
        bottleneck_size : int, default=32
            Number of output channels in the bottleneck layer.
        kernel_size : int, default=41
            Base kernel size for the convolutional layers.
        """
        super().__init__()
        self.use_residual = use_residual
        self.depth = depth

        in_channels = num_channels
        res_channels = in_channels
        layers = []
        shortcut_layers = []
        for d in range(depth):
            # Add residual connection every 3 layers
            if use_residual and d % 3 == 2:
                shortcut_layers.append(ResidualBlock(res_channels, in_channels))
                res_channels = in_channels
            else:
                shortcut_layers.append(None)

            layers.append(
                InceptionModule(
                    in_channels=in_channels,
                    num_filters=num_filters,
                    bottleneck_size=bottleneck_size,
                    kernel_size=kernel_size,
                    use_bottleneck=use_bottleneck,
                )
            )
            in_channels = num_filters * 4  # Update in_channels after concatenation

        self.inception_blocks = nn.ModuleList(layers)
        self.shortcut_layers = nn.ModuleList(shortcut_layers)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # Input shape: (num_samples, num_channels, num_timepoints)
        x_shortcut = x
        for layer, shortcut in zip(self.inception_blocks, self.shortcut_layers):
            if shortcut is not None:
                x = self.relu(x + shortcut(x_shortcut))
                x = layer(x)
                x_shortcut = x
            else:
                x = layer(x)
        x = self.global_avg_pool(x).squeeze(-1)  # Global average pooling
        x = self.fc(x)  # Fully connected layer for classification
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Residual Block to add a shortcut connection.
        """
        super().__init__()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # Adjust dimensions if the number of input and output channels differ
            self.shortcut = nn.Sequential(
                Conv1dSame(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return self.shortcut(x)


class Conv1dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        # Compute dynamic padding
        padding_needed = max(
            0,
            (self.stride - 1) * x.shape[-1]
            + self.dilation * (self.kernel_size - 1)
            + 1
            - self.stride,
        )
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left

        # Apply padding
        x = nn.functional.pad(x, (padding_left, padding_right))
        return self.conv(x)


class TimeSeriesSiameseNetwork(nn.Module):
    def __init__(self, input_channels, embedding_size):
        super().__init__()
        self.InceptionModel = InceptionModel(
            num_channels=input_channels,
            num_classes=embedding_size,
            depth=6,
            kernel_size=41,
        )

    def forward_once(self, x):
        output = self.InceptionModel(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class TimeSeriesTripletNetwork(nn.Module):
    def __init__(self, input_channels, embedding_size):
        super().__init__()
        self.InceptionModel = InceptionModel(
            num_channels=input_channels,
            num_classes=embedding_size,
            depth=6,
            kernel_size=41,
        )

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output


# Example Usage
if __name__ == "__main__":
    # Example input shape: (batch_size, num_channels, num_timepoints)
    num_channels = 3  # 3 input channels
    num_timepoints = 128  # 128 time steps
    num_classes = 10  # 10 output classes

    # Create the model
    model = InceptionModel(
        num_channels=num_channels, num_classes=num_classes, depth=6, kernel_size=41
    )
    print(model)

    # Example input tensor
    x = torch.randn(32, num_channels, num_timepoints)  # Batch size of 32
    y = model(x)  # Forward pass
    print(y.shape)  # Output shape: (32, num_classes)
