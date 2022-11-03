import torch
from torch import nn


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将通道数ch调整为divisor的整数倍（方便并行计算等）
    :param ch: channels, 需要调整的原始通道数量
    :param divisor: 调整为divisor的整数倍
    :param min_ch: 调整后的最小通道数
    :return: 调整后的通道数
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保向下取整时，不会减少超过 10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1, norm_layer=None):
        """
        卷积层+BatchNormalization+ReLU6
        :param in_channel: 输入特征矩阵深度
        :param out_channel: 输出特征矩阵深度
        :param kernel_size: 卷积核大小
        :param stride: 步距
        :param groups: groups=1则为普通卷积；groups=in_channel 则为 DW 卷积
        :param norm_layer: 是否需要FrozenBatchNorm2d
        """
        # 保证特征图宽高不变：3 -> 1, 1 -> 0
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, norm_layer=None):
        """
        倒残差结构的实现
        :param in_channel: 输入特征矩阵的通道数
        :param out_channel: 输出特征矩阵的通道数
        :param stride: 每个block中，3x3卷积核的步距
        :param expand_ratio: 中间层增加的通道数的倍率
        :param norm_layer: 是否需要FrozenBatchNorm2d
        """
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # 只有当stride=1且输入通道数等于输出通道数时，才进行残差连接
        self.use_short_cut = stride == 1 and in_channel == out_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        layers = []
        # 第一层bottleneck的expand_ratio=1，不需要第一个 1x1 卷积层
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv, linear activate method
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            norm_layer(out_channel)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_short_cut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, weights_path=None, norm_layer=None):
        """
        :param num_classes: 分类的类数
        :param alpha: 超参数，选取卷积核个数的倍率
        :param round_nearest: 需要将channel数变为这个数的倍数，方便并行计算等
        :param weights_path: 预训练权重路径，如果没有则进行初始化
        :param norm_layer: 是否需要FrozenBatchNorm2d
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        inverted_residual_setting = [
            # t, c, n, s
            # (扩展因子expand_ratio, 输出的channel, 模块重复的次数, block第一层的步距stride)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []
        # conv1 layer, 从 224x224x3 的图片到第一层特征 112x112x32
        features.append(ConvBNReLU(3, input_channel, stride=2))
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # 最后几层
        features.append(ConvBNReLU(input_channel, last_channel, 1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        if weights_path is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)


if __name__ == '__main__':
    # net = MobileNetV2(weights_path='../weights/mobilenet_v2.pth')
    # print(net)
    net = MobileNetV2(num_classes=5)
    print(net)
    pre_weights = torch.load('../weights/mobilenet_v2.pth')
    pre_dict = {k: v for k, v in pre_weights.items() if 'classifier' not in k}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    print('missing_keys: ', missing_keys)
    print('unexpected_keys: ', unexpected_keys)