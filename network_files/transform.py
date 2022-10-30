import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .image_list import ImageList


class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_min = image_mean
        self.image_std = image_std

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n'
        format_string += '{0}Normalize(mean={1}, std={2})'.format(
            _indent, self.image_min, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(
            _indent, self.min_size, self.max_size
        )
        format_string += '\n)'
        return format_string

    def normalize(self, image):
        dtype, device = image.detype, image.device
        mean = torch.as_tensor(self.image_min, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # image 的 shape 为 [channel, height, width]，其中 channel=3
        # mean 和 std 的 shape 为 [3],需要转化为 [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]

        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        image = self.resize_image(image, size, self.max_size)
        if target is None:
            return image, target
        bbox = target['boxes']
        bbox = self.resize_boxes(bbox, [h, w], image[-2:])
        target['boxes'] = bbox
        return image, target
    
    def batch_images(self, images, size_divisible=42):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch后返回（batch中每个image的shape是相同的）
        :param images: 输入的一batch图片
        :param size_divisible: 将图像的高和宽调整到这个数的正数倍(向上取整至)
        :return: 打包成一个batch的tensor
        """
        
        # 分别计算batch中图片的最大channel, height, width
        img_shapes = [list(img.shape) for img in images]
        max_size = self.max_by_axis(img_shapes)
        # 向上取整到stride的整数倍
        stride = float(size_divisible)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
        # [batch_size, channel, height, width]
        batch_shape = [len(images)] + max_size
        # 创建shape为batch_shape且全0的tensor
        batch_imgs = images[0].new_zeros(batch_shape)
        for img, pad_img in zip(images, batch_imgs):
            # copy_: 深拷贝，将img的值写入pad_img所在的内存中
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)
        return batch_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        :param result: 网络的预测结果, len(result) == batch_size，其中包含bbox和对应的类别
        :param image_shapes: 图像预处理缩放后的尺寸
        :param original_image_sizes: 图像的原始尺寸
        :return: 还原回原尺寸图片的结果，用于绘制bbox
        """
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred['boxes']
            boxes = self.resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes
        return result

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        # 记录 resize 后的图像尺寸
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    @staticmethod
    def torch_choice(k):
        # type: (List[int]) -> int
        # 从 list 中随机选取一个元素并返回
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    @staticmethod
    def resize_image(image, self_min_size, self_max_size):
        # type: (Tensor, float, float) -> Tensor
        # 首先尝试将图片的短边缩放到 self_min_size
        # 如果此时长边超过 self_max_size, 则重新将长边缩放至 self_max_size
        im_shape = image.shape[-2:]
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        scale_factor = self_min_size / min_size
        if max_size * scale_factor > self_max_size:
            scale_factor = self_max_size / max_size
        # 利用双线性插值缩放图片
        # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
        # bilinear只支持4D Tensor
        image = F.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear',
            align_corners=False, recompute_scale_factor=True
        )[0]
        return image

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        # type: (Tensor, List[int], List[int]) -> Tensor
        """
        将图像中的框从 original_size 缩放至 new_size
        :param boxes: 需要缩放的框，Tensor.shape(N, 4)
        :param original_size: List[Tensor.shape(2, )], len() = N
        :param new_size: : List[Tensor.shape(2, )], len() = N
        :return: 缩放后的boxes
        """
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratios_height, ratios_width = ratios
        # boxes: [minibatch, 4] 表示图片中的真实框数目，和框的坐标
        # 这里需要将它们的第一维展开
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    @staticmethod
    def max_by_axis(the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes




