# backbone 网络简要笔记

## MobileNet 系列

论文：MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Application

- 相比 VGG16 准确率减少 $0.9%$，但模型参数只有 VGG16 的 1/32;
- 亮点：增加了 Depthwise Convolution, 和超参数 $\alpha,\beta$

### MobileNet V1

> **Depthwise Separable Convolution（深度可分卷积）**
>
> - 传统卷积：卷积核通道数=输入特征矩阵通道数；卷积核个数=输出特征矩阵通道数
> - 深度可分卷积由下面两部分组成：
>   - DW卷积（Depthwise Conv）：每个卷积核的通道数均为 $1$；卷积核个数=输入特征矩阵通道数=输出特征矩阵通道数
>   - PW卷积（Pointwise Conv）：即 $1\times 1$ 卷积
> - DW+PW可以达到和普通卷积相同的输出，且计算量仅为普通卷积的 1/8 到 1/9
> - $\alpha$ 参数：MobileNet Width Multiplier，表示卷积核个数的倍率
> - $\beta$ 参数：MobileNet Resolution，表示图片分辨率，例如从224缩放至192

### MobileNet V2

论文：MobileNetV2: Inverted Residuals and Linear Bottlenecks

- MobileNet V1 中，DW卷积核参数大部分为0
- MobileNet V2 的准确率更高、网络更小
- 亮点：Inverted Residuals（倒残差结构），Linear Bottlenecks

> - Residual block: $1\times 1$ 卷积降维，$3\times 3$ 卷积，$1\times 1$ 卷积升维
> - Inverted Residual block:
>   - $1\times 1$ 卷积升维，$3\times 3$ DW卷积，$1\times 1$ 卷积降维
>   - 使用 `ReLU6` 激活函数：$y=\text{ReLU6}(x)=\min(\max(x,0),6)$
>   - 当 `stride=1` 且输入特征矩阵和输出特征矩阵 shape 相同时，才有 shortcut 连接
> - Linear Bottlenecks:
>   - 每个 Inverted Residual block 的最后一层使用线性激活函数
>   - 可以理解为ReLU对低维特征信息有大量损失，而倒残差结构的输出特征维度较小

### MobileNet V3

论文：Searching for MobileNetV3

- 更新Block（bneck）；
- 使用NAS搜索参数（Neural Architecture Search）
- 重新设计耗时层结构

![截屏2022-11-01 下午3.53.16](https://goatwu.oss-cn-beijing.aliyuncs.com/img/%E6%88%AA%E5%B1%8F2022-11-01%20%E4%B8%8B%E5%8D%883.53.16.png)

> **更新结构：**
>
> - 经过DW卷积后得到的特征矩阵，对每个channel进行pooling处理，得到 $C\times1\times1$ 的特征矩阵
> - 通过两个全连接层得到输出矩阵（本质上是一个注意力机制）
>   - 第一个全连接层映射到channel数的 1/4，ReLU激活函数
>   - 再通过全连接层得到和原来channel数相同的向量，hard-sigmoid激活函数
>   - 得到的向量作为channel维度的加权系数，得到输出矩阵
>
> **重新设计耗时层结构：**
>
> - 减少第一个卷积层的卷积核个数（32 to 16）
> - 精简 Last Stage
>
> **重新设计激活函数**
>
> - $\text{swish}\ x=x\times \sigma(x)$ 计算、求导较复杂；
>
> - 使用 hard-sigmoid激活函数和 hard-swish激活函数
>
> - $$
>   \text{h-sigmoid}[x]=\frac{\text{ReLU6}(x+3)}{6}\\
>   \text{h-swish}[x]=x\frac{\text{ReLU6}(x+3)}{6}
>   $$
