import torch
import math

class LoRALayer(torch.nn.Module):
    def __init__(self, original_layer, rank=16, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_channels
        out_features = original_layer.out_channels

        self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))

        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        orig_out = self.original_layer(x)
        lora_weight_matrix = self.lora_B @ self.lora_A
        lora_kernel = lora_weight_matrix.view(
            lora_weight_matrix.shape[0],
            lora_weight_matrix.shape[1],
            1, 1
        )
        lora_out = torch.nn.functional.conv2d(
            x, lora_kernel,
            stride=self.original_layer.stride,
            padding=0
        ) * self.scaling
        return orig_out + lora_out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'original_layer':
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return getattr(self.original_layer, name)