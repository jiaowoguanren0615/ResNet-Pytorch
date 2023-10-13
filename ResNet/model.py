import torch
import torch.nn as nn
from timm.models import register_model


class BasicBlock(nn.Module):

    expansion = 1
    """
    Represents the change of channel within each residual structure block \
    (the channels in each structure block are the same in ResNet18 and ResNet34)
    """

    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, **kwargs) -> None:
        super().__init__()
        self.C1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.B1 = nn.BatchNorm2d(num_features=out_channels)
        self.A1 = nn.ReLU(inplace=True)
        self.C2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.B2 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        if self.down_sample is not None:
            residual = self.down_sample(x)
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.C2(x)
        x = self.B2(x)
        y = x + residual
        y = self.A1(y)
        return y


class BottleNeck(nn.Module):

    expansion = 4
    """
    Represents the change of channels within each residual structure block 
    (the last channel in each structure block in ResNet50, ResNet101 and ResNet152 is 4 times the size of the previous two)
    """

    def __init__(self, in_channels, out_channels, down_sample=None, stride=1, groups=1, width_per_group=64) -> None:
        super().__init__()

        width = int(out_channels * (width_per_group / 64)) * groups

        self.C1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.B1 = nn.BatchNorm2d(num_features=width)
        self.A1 = nn.ReLU(inplace=True)
        self.C2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.B2 = nn.BatchNorm2d(num_features=width)
        self.C3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.B3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        if self.down_sample is not None:
            residual = self.down_sample(x)
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.C2(x)
        x = self.B2(x)
        x = self.A1(x)
        x = self.C3(x)
        x = self.B3(x)
        y = x + residual
        y = self.A1(y)
        return y


class ResNet(nn.Module):

    def __init__(self, block, block_list, num_classes, groups=1, width_per_group=64, distillation=None, **kwargs) -> None:
        super().__init__()  #block_list为存放每一层残差结构块堆叠次数的结构块

        assert len(block_list) == 4

        self.initial_filters = 64
        self.groups = groups
        self.width_per_group = width_per_group

        self.C1 = nn.Conv2d(3, self.initial_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.B1 = nn.BatchNorm2d(num_features=self.initial_filters)
        self.A1 = nn.ReLU(inplace=True)
        self.P1 = nn.MaxPool2d(3, 2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_list[0])
        self.layer2 = self._make_layer(block, 128, block_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_list[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        down_sample = None

        if stride != 1 or self.initial_filters != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.initial_filters, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=channel * block.expansion)
            )

        layers = []
        layers.append(block(self.initial_filters, channel, down_sample=down_sample, stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))

        self.initial_filters = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(
                block(self.initial_filters, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.P1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        y = self.fc(x)
        return y


@register_model
def resnet18(num_classes, **kwargs):
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return net


@register_model
def resnet34(num_classes, **kwargs):
    net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return net


@register_model
def resnet50(num_classes, **kwargs):
    net = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return net


@register_model
def resnet101(num_classes, **kwargs):
    net = ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    return net


@register_model
def resnet152(num_classes, **kwargs):
    net = ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    return net