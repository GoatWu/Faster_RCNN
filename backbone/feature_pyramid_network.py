from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.jit.annotations import Tuple, List, Dict


class LastLevelMaxpool(nn.Module):
    
    def forward(self, x, y, names):
        # type: (List(Tensor), List[Tensor], List[str]) -> Tuple[List[Tensor], List[str]]
        names.append('pool')
        x.append(F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return x, names


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        """
        从模型返回中间层的模块包装器。
        它有一个很强的假设：即模型的各个模块在 __init__ 中注册的顺序和在 forward 中
        正向传播的顺序相同。同时它只询问 model 的直属部分，例如 `model.feature1` 可以被返回，
        但是 `model.feature1.layer2` 不能被返回。
        :param model: 我们将提取特征的模型
        :param return_layers: (Dict[name, new_name])，
               一个字典，其中包含模块的名称，这些需要返回的中间层名称将作为字典的键返回，
               字典的值是这些中间层的新名称(用户可以指定)
        """
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构（简化结构）
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        """
        特征金字塔网络
        :param in_channels_list: (list[int]), 输入到 FPN 中的各个特征层通道数
        :param out_channels: (int), FPN 输出的通道数
        :param extra_blocks: 提供其它的操作层。这里是对最顶层特征进行maxpool得到P6
        """
        super(FeaturePyramidNetwork, self).__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
        self.extra_blocks = extra_blocks

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        计算得到 FPN 的一组输出结果
        :param x: (OrderedDict[Tensor]), 一组输入的特征图
        :return: (OrderedDict[Tensor])， 一组输出的特征图，其顺序从底层特征往上。
        """
        names = list(x.keys())
        x = list(x.values())
        # 将resnet layer4的channel调整到指定的out_channels
        last_inner = self.inner_blocks[-1](x[-1])
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        results.append(self.layer_blocks[-1](last_inner))
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            # 获取这一层特征的 h, w
            feat_shape = inner_lateral.shape[-2:]
            # 使用线性插值的方式，将上一层进行上采样至这一层的 h, w
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode='nearest')
            # 将上一层上采样的结果叠加至这一层，得到中间层输出
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class BackboneWithFPN(nn.Module):
    
    def __init__(self, backbone: nn.Module, return_layers=None, in_channels_list=None,
                 out_channels=256, extra_blocks=None, re_getter=True):
        super(BackboneWithFPN, self).__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxpool()
        if re_getter is True:
            assert return_layers is not None
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x
