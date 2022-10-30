from typing import Optional, List, Dict, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from . import det_utils
from . import boxes as box_ops


class RoIHeads(nn.Module):

    def __init__(self,
                 box_roi_pool,  # RoIAligned
                 box_head,      # TwoMLPHead
                 box_predictor, # FastRCNNPredictor
                 fg_iou_thresh, bg_iou_thresh,  # 0.5  0.5
                 batch_size_per_image, positive_fraction,   # 512  0.25
                 bbox_reg_weights,      # None
                 score_thresh,          # 0.055
                 nms_thresh,            # 0.5
                 detection_per_img):    # 100
        super(RoIHeads, self).__init__()
        self.box_similarity = box_ops.box_iou
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=False
        )
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.head = box_head
        self.box_predictor = box_predictor
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    @staticmethod
    def check_targets(targets):
        assert targets is not None
        assert all(['boxes' in t for t in targets])
        assert all(['labels' in t for t in targets])

    @staticmethod
    def add_gt_proposals(proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        :param proposals: 一个batch中每张图像rpn预测的boxes
        :param gt_boxes: 一个batch中每张图像对应的真实目标边界框
        :return: 一个列表，其中每个元素是一个拼接之后的Tensor，
                 shape: (post_nms_top_n+num_gt_boxes, 4)
        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        :param proposals: List[Tensor]
        :param gt_boxes: List[Tensor]
        :param gt_labels: List[Tensor]
        :return: 两个列表matched_idxs, labels，分别表示每个proposal所匹配的真实框标号，以及所属类别
        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1，舍弃为-2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        """
        获取每张图片中，采样到用于训练的样本anchors编号
        :param labels: [(num_anchors_per_img) * batch_size]
        :return: [(num_anchors_sampled) * batch_size]
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        :param proposals: RPN给出的预测框
        :param targets: 实现标注好的真实框及其标签
        :return: proposals, labels, regression_targets: 三个列表，
                 预测框      对应的标签     对应的回归参数
        """
        self.check_targets(targets)
        dtype = proposals[0].dtype
        device = proposals[0].device
        # 获取标注好的bbox信息和labels信息
        gt_boxes = [t['boxes'].to(dtype) for t in targets]
        gt_labels = [t['labels'] for t in targets]
        # 将gt_box拼接到proposals后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            # 获取对应正负样本的真实框坐标
            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    @staticmethod
    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        :param class_logits: (num_proposal_batch, num_classes)，batch中每个proposal对应每个类别的概率
        :param box_regression: (num_proposal_batch, num_classes*4)，
               表示batch中每个框到每一类真实框的回归参数，但是只有一类（也即真实框所属类）会被用到
        :param labels: [(num_proposal_image)]，每张图片中每个proposal对应的标签
        :param regression_targets: [(num_proposal_image, 4)]，每张图片中每个proposal对应回归参数
        :return:
        """
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        # 计算类别损失
        classification_loss = F.cross_entropy(class_logits, labels)
        # 返回标签类别大于0的索引
        sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
        # 返回标签类别大于0位置的类别信息
        labels_pos = labels[sampled_pos_inds_subset]
        # shape=(num_proposal_batch, num_classes)
        N, num_classes = class_logits.shape
        # (num_proposal_batch, num_classes*4) -> (num_proposal_batch, num_classes, 4)
        box_regression = box_regression.reshape(N, -1, 4)
        # 计算边界框损失信息
        box_loss = det_utils.smooth_l1_loss(
            # 先取出所有选中的正样本，再取得其对应其真实框的标签
            box_regression(sampled_pos_inds_subset, labels_pos),
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9, size_average=False,
        ) / labels.numel()
        return classification_loss, box_loss

    def postprocess_detections(self,
                               class_logits,  # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,  # type: List[Tensor]
                               image_shapes  # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """

        :param class_logits: (num_proposal_batch, num_classes)，batch中每个proposal对应每个类别的概率
        :param box_regression: (num_proposal_batch, num_classes*4)，
               表示batch中每个框到每一类真实框的回归参数，但是只有一类（也即真实框所属类）会被用到
        :param proposals: List[(num_proposal_image, 4)]，每张图片中rpn给出的的proposals坐标
        :param image_shapes: List[(2,)]，每张图片transform前的原始尺寸
        :return:
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        # 获取每张图像的预测bbox数量
        boxes_per_image = [bboxes_in_image.shape[0] for bboxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)
        # 分裂出每张图片的结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        all_boxes, all_scores, all_labels = [], [], []
        # 遍历每张图像预测信息
        # (proposals_per_img, num_classes, 4), (proposals_per_img, num_classes), (2,)
        for boxes, scores, image_shapes in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shapes)
            # 得到每个预测框的类别
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # 移除索引为0的所有信息（0代表背景），同时合并成一个大batch
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)
            # 移除低概率目标，self.scores_thresh=0.05
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            # 这里是对每个类别做nms处理
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]]) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        :param features: Dict[str, Tensor]，图像通过backbone后得到的
        :param proposals: List[Tensor[N, 4]]，RPN生成的proposals
        :param image_shapes: List[Tuple[H, W]]，记录输入图片缩放前的原始尺寸
        :param targets: List[Dict]，图像的标注信息
        :return:
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t['boxes'] in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels, regression_targets = None, None
        # box_features_shape: [num_proposals_batch, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # box_features_shape: [num_proposals_batch, representation_size]
        box_features = self.head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = self.fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        'boxes': boxes[i],
                        'labels': labels[i],
                        'scores': scores[i],
                    }
                )
        return result, losses
