import torch
import math
from typing import List, Tuple
from torch import Tensor


class BoxCoder:
    """
    将一组bbox编码和解码，用于rpn训练中的回归
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors以及对应的gt，计算regression参数。
        设 N 为batch中图像的数目，M 为每张图片中 anchors 的数目：
        :param reference_boxes: 每个anchor所对应的gt_box，len=N, shape=(M, 4)
        :param proposals: 每个anchor本身的坐标，len=N, shape=(M, 4)
        :return:
        """
        # 统计出每张图片中的 anchors 数目
        boxes_per_image = [len(b) for b in reference_boxes]
        # 将每个batch中每张图片的 gt_boxes 进行拼接
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        求得两组框之间的回归参数
        :param reference_boxes: Tensor.shape=(N*M, 4)
        :param proposals: Tensor.shape=(N*M, 4)
        :return:
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.dtype
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = self.encode_boxes(reference_boxes, proposals, weights)
        return targets

    @staticmethod
    def encode_boxes(reference_boxes, proposals, weights):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        求得两组框之间的回归参数
        :param reference_boxes: Tensor.shape=(N*M, 4)
        :param proposals: Tensor.shape=(N*M, 4)
        :param weights:
        :return: 编码后的Tensor，shape=(N*M, 1, 4)
        """
        # anchors的宽高，以及中心坐标
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights
        # gt_boxes 的宽高、中心坐标
        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        targets_dx = weights[0] * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = weights[1] * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = weights[2] * torch.log(gt_widths / ex_widths)
        targets_dh = weights[3] * torch.log(gt_heights / ex_heights)
        targets = torch.cat((targets_dx[:, None], targets_dy[:, None],
                             targets_dw[:, None], targets_dh[:, None]), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """
        将每个batch生成的锚框和回归出来的偏移量进行合并。设batch_size为N:
        tot_num_anchors_image = sum(w * h * anchors_per_pos) for every feature maps
        :param rel_codes: 回归参数。shape: (N * tot_num_anchors_image, 4)
        :param boxes: 生成的bbox。shape: [(tot_num_anchors_level, 4)] * N
        :return:
        """
        concat_boxes = torch.cat(boxes, dim=0)  # (N * tot_num_anchors_level, 4)
        box_sum = concat_boxes.shape[0]
        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )
        if box_sum > 0:
            # (box_sum, 4) -> (box_sum, 1, 4)
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)
        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        # anchors 的中心坐标
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5]
        # 使用 ::4 表示间隔为4的采样。shape: (box_sum, 1)
        dx = rel_codes[:, 0::4] / wx  # 预测anchors的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy  # 预测anchors的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww  # 预测anchors的宽度回归参数
        dh = rel_codes[:, 3::4] / wh  # 预测anchors的高度回归参数
        # 限制将过大的值送入 torch.exp(), 对dw和dh设置上限
        # self.bbox_xform_clip=math.log(1000. / 16)   4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        # 获得修正后的bbox中心坐标与宽高
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = widths[:, None] * torch.exp(dw)
        pred_h = heights[:, None] * torch.exp(dh)
        # xmin, ymin, xmax, ymax
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.cat((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=1)
        return pred_boxes


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算 anchors 与每个 gt_box 的最大 iou，并记录索引
        :param match_quality_matrix: Tensor([N, M])
        :return: matches: shape(M), 表示每个anchor所对于的真实框id，或者对应背景/对应丢弃
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        # 求每一列的最大值（也即每个 anchor 所对应的最大 gt_box）
        # matched_vals 为其数值，matches 为其索引
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        # 计算 iou 小于 low_threshold 的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算iou在low_threshold与high_threshold之间的索引值
        between_threshold = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold)
        # iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        # iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_threshold] = self.BETWEEN_THRESHOLDS
        # 剩余的还是原来的matches值，也即匹配的gt_box索引
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches(matches, all_matches, match_quality_matrix)
        return matches

    @staticmethod
    def set_low_quality_matches(matches, all_matches, match_quality_matrix):
        """
        有可能存在某些gt_boxes，没有被任何anchors匹配到（如：最高iou为0.65）
        因此需要对每个gt_boxes寻找与其iou最大的anchor, 将其也设置为正样本
        :param matches: Tensor(M), 每个anchor所匹配到的gt_box或者是背景/无用框
        :param all_matches: Tensor(M), 每个anchor所对应的最大iou的gt_boxes
        :param match_quality_matrix: Tensor(N, M), 原始的iou矩阵
        :return: None
        """
        # 对于每个 gt_box 寻找与其 iou 最大的 anchor
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # 寻找每个gt boxes与其iou最大的anchor索引，一个gt匹配到的最大iou可能有多个anchor
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )
        # Example gt_pred_pairs_of_highest_quality:
        # (Tensor([0,      1,      1,      2,      2,      3,      3,      4]),
        #  Tensor([39796,  32055,  32070,  39190,  40255,  40390,  41455,  45470]))
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image, positive_fraction):
        """
        :param batch_size_per_image: int类型，每张图片取多少个样本
        :param positive_fraction: float类型，每张图片中正样本的比例
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """

        :param matched_idxs:
        :return:
        """
        pos_idx = []
        neg_idx = []
        for matched_idx_per_img in matched_idxs:
            # = 1的为正样本, = 0的为负样本
            positive = torch.where(torch.eq(matched_idx_per_img, 1))[0]
            negative = torch.where(torch.eq(matched_idx_per_img, 0))[0]
            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # 如果正样本数量不足，则使用全部的正样本
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            # 随机选取指定数量的正负样本
            # torch.randperm(n): 返回 [0, n-1] 的一个随机排列
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            # 创建蒙版
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idx_per_img, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idx_per_img, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1
            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    计算边界框的回归损失
    :param input: 选取的正样本预测的边界框回归值，Tensor.shape = (num_choose_pos_anchors, 4)
    :param target: 真实边界框回归值，Tensor.shape = (num_choose_pos_anchors, 4)
    :param beta: 超参数
    :param size_average: 返回平均值/总和
    :return:
    """
    n = torch.abs(input - target)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
