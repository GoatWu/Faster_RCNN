import torch
from typing import Tuple
from torch import Tensor


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    返回两个bbox的交并比。每个bbox记录其左上与右下坐标 (x1, y1, x2, y2)。方法是：
    1. 求得两两之间，左上角得较大值
    2. 求得两两之间，右下角得较小值
    3. 两点所构成得矩形框即为交叉部分
    4. 如果两个矩形不相交，则 1 和 2 中求得得左上和右下不构成矩形
    :param boxes1: (Tensor[N, 4])
    :param boxes2: (Tensor[M, 4])
    :return: (Tensor[N, M]): 第一个Tensor的 N 个 bbox与
             第二个 Tensor 的 M 个 bbox 的交并比
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # 广播机制：(N, 1, 2) 和 (M, 2) 的运算结果为 (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    调整预测bbox的信息，使其坐标均在原图像内部
    :param boxes: (Tensor[len(top_n_idx), 4]) in (x1, y1, x2, y2)
    :param size: (Tuple[height, width]): 图片的原始尺寸（不包含填充的0）
    :return:  clipped_boxes (Tensor[len(top_n_idx), 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size
    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    移除宽高小于指定阈值的索引
    :param boxes: (Tensor[N, 4]) in (x1, y1, x2, y2)
    :param min_size: 最小尺寸
    :return: Tensor[k]: 宽高均大于 min_size 的索引 Tensor
    """
    # bboxes 的宽和高
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]
    return keep


def nms(boxes, scores, iou_threshold):
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    非极大值抑制处理，在整个batch中去掉重复的框。注意这里的batch指的是特征层，而非图片。
    :param boxes: Tensor[N, 4], 每张图片中的预测框
    :param scores: Tensor[N], 每个预测框的前景分数
    :param idxs: Tensor[N], 每个预测框属于哪个特征层
    :param iou_threshold: float, iou大于此分数的框被抑制
    :return: NMS处理后，剩下的预测框索引
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    # 将每一层的框坐标加上此最大值 * 层的idx，可以直接区分每个框属于哪层
    max_coordinate = boxes.max()
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    # 利用广播机制，为每个层生成一个偏移量，保证每层的boxes不会有重合
    offset = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offset[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
