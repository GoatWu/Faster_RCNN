from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .image_list import ImageList
from . import det_utils
from . import boxes as box_ops


class RPNHead(nn.Module):
    """
    在特征图上使用两个 1x1 的卷积，保持特征图的大小不变
    channel数分别为 k 和 4k，分别表示特征图上每个点生成的 k 个 anchor中
    每个 anchor 的前景概率、每个 anchor 的偏移回归参数
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3滑动窗口, 输入的channel=输出的channel
        # 一个细节：使用padding=1保证特征图的大小不变
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                              stride=1, padding=1)
        # 计算预测的目标分数（区分bbox对应前景还是背景）,这里为输出前景的分数
        self.cls_logits = nn.Conv2d(in_channels, num_anchors,
                                    kernel_size=1, stride=1)
        # 计算预测的目标的bbox regression参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
                                   kernel_size=1, stride=1)
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        # 在mobilenet中，x只包含一个特征层；resnet中则有多个，用list形式存储
        logits = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = aspect_ratios * len(sizes)
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}  # 存储生成anchors的坐标信息

    @staticmethod
    def generate_anchors(scales, aspect_ratios,
                         dtype=torch.float32,
                         device=torch.device('cpu')):
        """
        给定需要生成锚框的比例和原始高宽，给出一组中心坐标为 (0, 0) 的锚框
        :param scales: 锚框的原始高宽
        :param aspect_ratios: 锚框的比例
        :param dtype: 数据类型
        :param device: 锚框所在设备
        :return: 一组中心坐标为 (0, 0) 的锚框
        """
        # type: (List[int], List[float], torch.dtype, torch.device)
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # aspect_ratios 为 anchors 的高宽比，同时保证 anchors 的面积一致
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0 / h_ratios
        # 利用广播机制，使得列向量和行向量相乘，拓展成矩阵的对应相乘
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        # 生成的anchors模板都是以（0, 0）为中心的, shape [len(ratios)*len(scales), 4]
        # 每一行为一个 anchor
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        """
        在给定设备和类型后生成一组中心坐标为 (0, 0) 的锚框，
        并赋值给类自身的变量 self.cell_anchors
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            if cell_anchors[0].device == device:
                return
        # 根据提供的sizes和aspect_ratios生成anchors模板
        # anchors模板都是以(0, 0)为中心的anchor
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        计算预测特征图对应原始图像上的所有anchors的坐标
        :param grid_sizes: 预测特征矩阵的height和width
        :param strides: 预测特征矩阵上一步对应原始图像上的步距
        :return: 应原始图像上的所有anchors的坐标
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_h, grid_w = size
            stride_h, stride_w = stride
            device = base_anchors.device
            # 生成特征图上每个点在原图中对应的坐标
            shifts_x = torch.arange(0, grid_w, dtype=torch.float32, device=device)
            shifts_y = torch.arange(0, grid_h, dtype=torch.float32, device=device)
            # 第一个矩阵所有列都是 shifts_y, 第二个矩阵所有行都是 shifts_x
            # 对应起来就是特征图每个点在原图中的坐标
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            # 得到特征图中每个点在原图中对应锚框的中心坐标
            # 加上锚框模板后即可得到原图中的锚框
            # shape: [grid_w * grid_h, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)
            # 每个点对应多个锚框模板，因此变成三维后利用广播机制相加
            # shape: [grid_w*grid_h, aspect_ratio*scales, 4]
            shifts_anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchors.reshape(-1, 4))
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        将计算得到的所有anchors信息进行缓存
        :param grid_sizes: 各个特征图的尺寸
        :param strides: 各个特征图中一个cell对应的原图步长
        :return: 在特征图上得到的anchors
        """
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # 获取每个预测特征层的尺寸(height, width)
        grid_sizes = list([f_map.shape[-2:] for f_map in feature_maps])
        # 获取输入图像的height和width
        image_size = image_list.tensors.shape[-2:]
        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        # 计算特征层上的一步等于原始图像上的步长
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)]
                   for g in grid_sizes]
        # 根据提供的sizes和aspect_ratios生成anchors模板
        self.set_cell_anchors(dtype, device)
        # 计算（或读取）所有anchors在原图中的坐标
        # 得到一个list，其中每个元素为在每张特征图中得到的anchors坐标
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个 batch 中的每张图像
        for i in range(len(image_list.image_sizes)):
            # anchors_in_image = []
            # for anchors_per_feature_map in anchors_over_all_feature_maps:
            #     anchors_in_image.append(anchors_per_feature_map)
            # anchors.append(anchors_in_image)
            anchors.append(anchors_over_all_feature_maps)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有anchors信息
        anchors = [torch.cat(anchors_per_image)
                   for anchors_per_image in anchors]
        self._cache.clear()
        return anchors


class RegionProposalNetwork(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 # 存在大于此阈值的iou则视为前景 全部小于此iou的则视为背景
                 # 其余的锚框均被舍弃
                 fg_iou_thresh, bg_iou_thresh,
                 # rpn在计算损失时采用的正负样本总个数 其中正样本的比例
                 batch_size_per_image, positive_fraction,
                 # nms前每个特征图的锚框数 nms后输出的总proposal数
                 pre_nms_top_n, post_nms_top_n,
                 # nms处理时的阈值
                 nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # 训练中所用参数
        self.box_similarity = None
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # iou 大于 fg_iou_thresh(0.7) 时视为正样本
            bg_iou_thresh,  # iou 小于 bg_iou_thresh(0.3) 时视为负样本
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        # 验证中所用参数
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def pre_nms_top_n(self):
        return self._pre_nms_top_n['training' if self.training else 'testing']

    def post_nms_top_n(self):
        return self._post_nms_top_n['training' if self.training else 'testing']

    @staticmethod
    def permute_and_flatten(layer, N, A, C, H, W):
        # type: (Tensor, int, int, int, int, int) -> Tensor
        # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
        # view函数只能用于内存中连续存储的tensor，
        # permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
        # reshape则不需要依赖目标tensor是否在内存中是连续的
        layer = layer.view(N, A, C, H, W)
        # 调换tensor维度
        layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, A, C]
        layer = layer.reshape(N, -1, C)
        return layer

    def concat_box_prediction_layers(self, box_cls, box_regression):
        # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        box_cls_flattened = []
        box_regression_flattened = []
        for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
            # shape: [batch_size, anchors_per_pos * classes, height, width]
            N, AxC, H, W = box_cls_per_level.shape
            # shape: [batch_size, anchors_per_pos * 4, height, width]
            Ax4 = box_regression_per_level.shape[1]
            # anchors_num_per_position
            A = Ax4 // 4
            # classes_num, 这里为1
            C = AxC // A
            # [N, -1, C]
            box_cls_per_level = self.permute_and_flatten(box_cls_per_level, N, A, C, H, W)
            box_cls_flattened.append(box_cls_per_level)
            # [N, -1, C]
            box_regression_per_level = self.permute_and_flatten(box_cls_per_level, N, A, 4, H, W)
            box_regression_flattened.append(box_regression_per_level)
        # box_cls_flattened.shape: [(N, tot_num_anchors_feature_map, C)]
        # box_cls.shape: (N * tot_num_anchors_image, C)
        # box_regression.shape: (N * tot_num_anchors_image, 4)
        # 即将整个 batch 的 box_cls 放在一起
        box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
        box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
        return box_cls, box_regression

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        :param objectness: Tensor.shape(N, tot_num_anchors_image), 表示每个框的预测概率
        :param num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        :return: 返回 proposals: shape=(N, return_num_anchors)
        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 从第 1 维度进行分割，每段长度为 num_anchors_per_level，即可得到每个特征层上的预测概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            # pre_nms_top_n 个数的数值，以及它们的索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小的bbox，进行nms处理，根据预测概率获得post_nms_top_n个目标
        :param proposals: 预测的bbox坐标
        :param objectness: 预测的目标概率
        :param image_shapes: batch中每张图片的size信息
        :param num_anchors_per_level: 每个预测特征层上的anchors数目
        :return:
        """
        num_images = proposals.shape[0]  # batch_size
        device = proposals.device
        # 对于 fast rcnn 部分，proposals 是输入参数，不需要梯度
        objectness = objectness.detach()
        # (N * tot_num_anchors_image, 1) -> (N, tot_num_anchors_image)
        objectness = objectness.reshape(num_images, -1)
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        # [tensor(0, 0, 0, ...), tensor(1, 1, 1, ...), ...]
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, dim=0)
        # 将其变为和 objectness 相同维度的 tensor
        levels.expand_as(objectness)
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        # shape: (N, return_anchors_per_image)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)  # tensor([0...N])
        batch_idx = image_range[:, None]  # tensor([[0],\n    [1]\n, ...])
        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        # objectness.shape: (N, len(top_n_idx))
        # levels.shape: (N, len(top_n_idx))
        # proposals.shape: (N, len(top_n_idx), 4)
        objectness = objectness[batch_idx, top_n_idx]
        objectness_prob = torch.sigmoid(objectness)
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测bbox的信息，使其坐标均在原图像内部
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 移除小概率 boxes
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # nms: non-maximum suppression
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # 获取前 post_nms_top_n 个分数最高的框
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(keep)
            final_scores.append(scores)
        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchor所对应的ground truth，划分为正样本、负样本、废弃样本
        :param anchors: List(Tensor)，每张图片上生成的初始anchors
        :param targets: List[Dict[str, Tensor]]，列表中为每张图片的标注信息
        :return: labels, 列表中的每个元素为：每张图片中每个anchor对应前景/背景/丢弃，len=N, shape=(M,)
                 matched_gt_boxes, 列表中的每个元素为：每张图片中每个anchor对应的gt_box坐标，
                                   len=N, shape=(M, 4)
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图片中的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                # shape: (num_anchors_per_image, 4); (num_anchors_per_image)
                matched_gt_boxes_per_image = torch.zeros(
                    anchors_per_image.shape, dtype=torch.float32, device=device
                )
                labels_per_image = torch.zeros(
                    (anchors_per_image.shape[0],), dtype=torch.float32, device=device
                )
            else:
                # 计算 anchors 与真实 bbox 的 iou
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # 计算每个anchors与gt匹配iou最大的索引，并且-1表示背景、-2表示丢弃
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # 取出每个 anchor 所对应的 gt_box. 这里尽管负样本和丢弃样本有值，但是后面并不会被用到
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                # 得到正样本的蒙板（mask）
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # 得到负样本的蒙版
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                # 得到丢弃样本的蒙版
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），和bbox regression损失
        :param objectness: shape: (N * tot_num_anchors_image, 1)
                           表示训练出来的每个 anchor 是前景的概率
        :param pred_bbox_deltas: shape: (N * tot_num_anchors_image, 4)
                                表示训练出来的每个 anchor 的回归参数
        :param labels: List[Tensor: shape=(tot_num_anchors_image)]
                       表示每张图片的anchor所属类别，1为前景，0为背景，-1为丢弃
        :param regression_targets: shape: List[Tensor: shape=(tot_num_anchors_image)]
                                   表示每张图片的所有anchor的真实回归参数
        :return:
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中所有的正负样本List(Tensor)拼接在一起
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        # 将所有正负样本索引进行拼接
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()  # (x, 1) -> (x)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        # 计算边界框回归损失（只计算正样本）
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1. / 9, size_average=False,
        ) / (sampled_inds.numel())
        # 计算目标预测概率损失(方法内部会进行sigmoid处理)
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        return objectness_loss, box_loss

    def forward(self,
                images,  # type: ImageList
                features,  # type: Dict[str, Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # 将带标号的字典转化为列表
        # [shape=(batch_size, channel, height, width)]
        features = list(features.values())
        # 计算出每个特征层上的框前景概率，以及框的回归参数
        # objectness, pred_bbox_deltas 都是列表
        objectness, pred_bbox_deltas = self.head(features)
        # 生成一个batch图片所有anchors的信息
        anchors = self.anchor_generator(images, features)
        # batch_size
        num_images = len(anchors)
        # 计算每个特征层上对应的anchors数量
        # s: (channel, height, width), channel对应每个特征点生成的锚框数
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # 调整内部tensor格式以及shape
        objectness, pred_bbox_deltas = self.concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # (box_tot, 4) -> (batch_size, num_box_per_image, 4)
        proposals = proposals.view(num_images, -1, 4)
        # 移除小目标，nms处理，获取前 post_nms_top_n 个目标
        # shape: [(post_nms_top_n, 4)] * N, [post_nms_top_n] * N
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses

