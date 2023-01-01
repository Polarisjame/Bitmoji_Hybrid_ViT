import torch
from torch import nn



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, re_zero=False):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(4, out_channels)
        )  # out_size = in_size
        self.residual = shortcut
        self.re_zero = re_zero
        if re_zero:
            self.alpha = nn.Parameter(torch.zeros(1))
        self.activ = nn.GELU()

    def forward(self, x):
        left = self.layers(x)
        right = self.residual(x) if self.residual else x
        if self.re_zero:
            right = right * self.alpha
        out = left + right
        return self.activ(out)


class Res34(nn.Module):
    def __init__(self, args, in_channels, out_channels=None):
        super(Res34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=True),  # out_size = in_size / 2
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=False),
            nn.MaxPool2d(3, 2, 1)  # out_size = in_size//2 向上取整
        )
        self.re_zero = args.re_zero
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.hybrid = args.use_hybrid
        if not args.use_hybrid:
            self.layer3 = self._make_layer(256, 512, 6, stride=2)
            self.layer4 = self._make_layer(512, 512, 3, stride=2)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512, out_channels)
        else:
            self.layer3 = self._make_layer(256, out_channels, 9, stride=2)
        # self.softmax = nn.Softmax(1)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = [ResidualBlock(inchannel, outchannel, stride, shortcut, self.re_zero)]
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel, re_zero=self.re_zero))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.hybrid:
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        # x = self.softmax(x)
        return x


class CNN(nn.Module):
    def __init__(self, args, inchannel, outchannel):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, 5, 1, 0),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 32, 7, 2),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, 2),
        )
        self.fcs = nn.Sequential(
            nn.Linear(32 * 12 * 12, 512),
            nn.GELU(),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, outchannel)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 32 * 12 * 12)
        x = self.fcs(x)
        return x
