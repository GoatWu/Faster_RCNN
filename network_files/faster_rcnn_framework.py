from typing import Tuple, List, Dict, Optional
from collections import OrderedDict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from .roi_head import RoIHeads


class FasterRCNNBase(nn.Module):

    def __init__(self, backbone, rpn, roi_heads, transform):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]

        if self.training and targets is None:
            raise ValueError('In training mode, targets should not be None.')
        if self.training:
            for target in targets:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))
        # 记录原图像的尺寸。这样经过 transform 后，得到的 bbox 可以映射回6原图
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]  # (channel, width, height)
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)  # 对图像进行预处理
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):  # 对于resnet，则直接传入一个有序字典
            features = OrderedDict([('0', features)])  # 表示只在一层特征图上进行预测
        # 将特征以及标注信息传入rpn中，得到建议框
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 将候选框和标注传入faster rcnn的后半部分
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # 将网络的预测结果进行后处理（bboxes还原到原图片上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections


class FasterRCNN(FasterRCNNBase):

    def __init__(self, backbone, num_classes=None,
                 min_size=600, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 rpn_anchor_generator=None, rpn_head=None,
                 # 带有RPN的网络有多个预测特征层，每层在nms前保留2000个，通过nms后总共保留2000个
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn在nms处理前保留的proposal数（根据score）
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时的iou阈值
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # 存在与Truth大于0.7的即为正样本，全部小于0.3的即为负样本
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # 每张图采样anchor数，其中正样本的比例
                 rpn_score_thresh=0.0,
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低概率目标       faster rcnn中进行nms处理的阈值  对预测结果根据score排序取前100
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )
        # assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels  # 特征向量的channels
        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        # 较浅层的特征层感受野小，识别较小的对象；较深的特征层感受野大，识别大的对象
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_localtion()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_test,
                                 testing=rpn_pre_nms_top_n_train)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                                  testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层上进行roi pooling
                output_size=[7, 7],  # 输出的特征图尺寸
                sampling_ratio=2  # 采样率
            )
        # faster rcnn中roi pooling后，展平处理两个MLP部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认为7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )
        # 在box_head输出上预测得分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes
            )
        # 将上面几个部分结合
        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh, # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img  # 0.05 0.5 100
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        """
        :param in_channels: 输入通道数
        :param representation_size: 中间层和输出层的通道数
        """
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # x: (batch_size*num_proposals, channels, H, W)
        # num_proposals为每张图片选取的proposals数，
        # channels为backbone/roi_pooling的输出通道数
        # H 和 W 为 ROI_Pooling 输出的特征图高宽，一般为 7x7
        # in_channels = num_proposals * channels * H * W
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        # x: (batch_size*num_proposals, representation_size)
        # representation_size 为 TwoMLPHead 的输出 channel 数。
        x = x.flatten(start_dim=1)  # 不起作用
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas
