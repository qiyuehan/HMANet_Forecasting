import torch
import torch.nn as nn
from utils.deformable import Def_Conv

class Multi_Def_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Multi_Def_Conv, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.conv2D = nn.Conv2d(in_channels*3, in_channels, 3,1,1)
        for dilation in dilation_rates:
            self.conv_layers.append(
                    Def_Conv(in_channels, in_channels, kernel_size=3, dilation_rate=dilation)  # Deformable Conv
                )

    def forward(self, x):
        output = []
        for conv_layer in self.conv_layers:
            output.append(conv_layer(x))
        cat_out = torch.cat(output, dim=1)
        output = self.conv2D(cat_out)
        return output






