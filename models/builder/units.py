"""
UNITS Module.
"""

def get_conv_unit(in_channels, out_channels):
    conv = torch.nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=3, stride=1, padding='same', dilation=1, groups=1,
    )
    batch_norm = torch.nn.BatchNorm2d(num_features=out_channels)
    relu = torch.nn.ReLU(inplace=True)
    return torch.nn.Sequential(
        conv, batch_norm, relu,
    )
