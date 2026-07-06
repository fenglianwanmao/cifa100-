
import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差基本块：两个 3x3 卷积 
class BasicBlock(nn.Module):
    """Basic residual block for CIFAR (similar to original TF version)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# CIFAR 特化 ResNet：32x32 输入，三阶段残差，通道 16→32→64
class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=100, num_blocks_list=(3, 3, 3)):
        super().__init__()
        self.in_channels = 16

        # 首层卷积：不做 7x7 和大池化，适配小图
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 三个残差阶段，后两阶段用 stride=2 下采样
        self.layer1 = self._make_layer(16, num_blocks_list[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks_list[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks_list[2], stride=2)

        # 全局池化 + 全连接分类头（粗 20 类 / 细 100 类由 num_classes 决定）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    # 堆叠多个 BasicBlock 组成一个 stage
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 创建 ResNet 模型，num_classes 随粗/细标签变化（20 或 100）
def get_model(name, num_classes):
    name = name.lower()
    if name in ('resnet', 'resnet_cifar'):
        return ResNetCIFAR(num_classes)
    raise ValueError(f"Unknown model: {name}. Use 'resnet'.")


if __name__ == "__main__":
    model = get_model('resnet', 100)
    print("ResNet params:", sum(p.numel() for p in model.parameters()))
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("Output shape:", out.shape)