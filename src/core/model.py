import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None, d: int = 1):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.path_a = nn.Sequential(
            ConvBNReLU(channels, channels, k=3, d=1),
            ConvBNReLU(channels, channels, k=3, d=2),
        )
        self.path_b = nn.Sequential(
            ConvBNReLU(channels, channels, k=1),
            ConvBNReLU(channels, channels, k=3, d=1),
        )
        self.fuse = ConvBNReLU(channels * 2, channels, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.path_a(x)
        b = self.path_b(x)
        y = torch.cat([a, b], dim=1)
        y = self.fuse(y)
        return x + y


class Baccarat2DCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, k=3),
            ConvBNReLU(64, 128, k=3),
        )

        # Pattern blocks
        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)
        self.down1 = ConvBNReLU(128, 256, k=3, s=2)
        self.block3 = ResidualBlock(256)
        self.block4 = ResidualBlock(256)

        # Extra
        self.down2 = ConvBNReLU(256, 512, k=3, s=2)
        self.block5 = ResidualBlock(512)

        # Head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.down1(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.down2(x)
        x = self.block5(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def get_model_info(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print("BACCARAT 2D CNN")
        print("=" * 60)
        print("Input: 3 x 6 x 12")
        print("Architecture: stem -> 2xRes(128) -> down -> 2xRes(256) -> down -> Res(512) -> FC")
        print(f"Params: total={total:,}, trainable={trainable:,}")

def create_model(device: str | torch.device = 'auto') -> Baccarat2DCNN:
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    model = Baccarat2DCNN(num_classes=3)
    model.to(device)
    return model

