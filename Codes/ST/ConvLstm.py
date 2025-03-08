import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        bias: bool,
        frame_size: tuple,
    ):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.frame_size = frame_size

        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        self.register_parameter(
            "W_ci", nn.Parameter(torch.zeros(self.hidden_dim, *self.frame_size))
        )
        self.register_parameter(
            "W_cf", nn.Parameter(torch.zeros(self.hidden_dim, *self.frame_size))
        )
        self.register_parameter(
            "W_co", nn.Parameter(torch.zeros(self.hidden_dim, *self.frame_size))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights for the ConvLSTMCell."""
        # Initialize convolutional weights with Xavier/Glorot
        nn.init.xavier_uniform_(self.conv.weight)

        # Initialize biases if they exist
        if self.bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):

        combined = torch.cat([x, h_prev], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i + self.W_ci * c_prev)
        f = torch.sigmoid(cc_f + self.W_cf * c_prev)
        c_next = f * c_prev + i * torch.tanh(cc_g)
        o = torch.sigmoid(cc_o + self.W_co * c_next)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM_Layer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        bias: bool,
        frame_size: tuple,
    ):

        super(ConvLSTM_Layer, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.frame_size = frame_size

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            frame_size=self.frame_size,
        )

    def forward(self, X: torch.Tensor):
        device = X.device
        b, _, t, h, w = X.size()

        # Initialize output
        output = torch.zeros(b, self.hidden_dim, t, h, w).to(device)

        # Initialize Hidden State
        H = torch.zeros(b, self.hidden_dim, h, w).to(device)

        # Initialize Cell Input
        C = torch.zeros(b, self.hidden_dim, h, w).to(device)

        # Unroll over time steps
        for time_step in range(t):

            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


class ConvLSTM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_dim: int,
        kernel_size: tuple,
        stride: int,
        padding: int,
        bias: bool,
        frame_size: tuple,
    ):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.frame_size = frame_size
        self.module = nn.Sequential()

        self.module.add_module(
            f"ConvLSTM_{1}",
            ConvLSTM_Layer(
                self.in_channels,
                self.hidden_dim,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias,
                self.frame_size,
            ),
        )
        self.module.add_module(f"batchnorm_{1}", nn.BatchNorm3d(hidden_dim))
        for i in range(2, self.num_layers + 1):
            self.module.add_module(
                f"ConvLSTM_{i}",
                ConvLSTM_Layer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.bias,
                    self.frame_size,
                ),
            )
            self.module.add_module(f"batchnorm_{i}", nn.BatchNorm3d(hidden_dim))

        # reshape the output
        self.final_conv = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for the ConvLSTM model."""
        # Initialize final conv layer
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.bias:
            nn.init.zeros_(self.final_conv.bias)

        # Initialize weights in BatchNorm layers
        for name, module in self.module.named_modules():
            if isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, X: torch.Tensor):
        out = self.module(X)
        return torch.tanh(self.final_conv(out[:, :, -1]).unsqueeze(2))


if __name__ == "__main__":
    model = ConvLSTM(
        num_layers=2,
        in_channels=1,
        hidden_dim=64,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=True,
        frame_size=(30, 30),
    )
    x = torch.randn(2, 1, 5, 30, 30)
    print(model(x).shape)