import os
import datetime
import torch
import torch.utils.data
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups


def create_model(num_classes):
    backbone = MobileNetV2(weights_path='../weights/mobilenet_v2.pth').features
    backbone.out_channels = 1280    # 对应backbone输出特征矩阵的channels
    anchor_generator = AnchorsGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0))
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=[7, 7],
        sampling_ratio=2
    )
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


def main(root):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device training.'.format(device.type))
    results_files = 'results{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    checkpoint_root = os.path.join(root, 'save_checkpoints/MobileNetV2')
    result_files = os.path.join(checkpoint_root, results_files)
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root)
    data_transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        'val': transforms.Compose([transforms.ToTensor()])
    }
    VOC_root = root
    aspect_ratio_group_factor = 3
    batch_size = 8
    amp = False     # 是否使用混合精度训练，需要GPU支持
    if os.path.exists(os.path.join(VOC_root, 'VOCdevkit')) is False:
        raise FileNotFoundError('VOCdevkit does not exist in path: "{}".'.format(VOC_root))
    train_dataset = VOCDataSet(VOC_root, '2012', data_transform['train'], 'train.txt')
    train_sampler, train_batch_sampler = None, None
    # 是否按照图片高宽比采样图片组成batch
    # 是的话能减小训练时所需GPU显存，默认使用
    if aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % num_workers)
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=num_workers,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=num_workers,
                                                        collate_fn=train_dataset.collate_fn)
    val_dataset = VOCDataSet(VOC_root, '2012', data_transform['val'], 'val.txt')
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=num_workers,
                                                  collate_fn=val_dataset.collate_fn)
    model = create_model(num_classes=21)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    train_loss = []
    learning_rate = []
    val_map = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  first frozen backbone and train 5 epochs                    #
    #  首先冻结前置特征提取网络权重(backbone), 训练rpn以及最终预测网络部分  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for param in model.backbone.parameters():
        param.requires_grad = False
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=0.0005)
    init_epochs = 5
    for epoch in range(init_epochs):
        pass
        # TODO: utils.train_one_epoch
        # mean_loss, lr = utils.train_one_epoch(
        #     model, optimizer, train_data_loader,
        #     device, epoch, print_freq=50, warmup=True, scaler=scaler
        # )
        # train_loss.append(mean_loss.item())
        # learning_rate.append(lr)
        # # 在验证集上验证
        # coco_info = utils.evaluate(model, val_data_loader, device=device)
        # # 保存到txt
        # with open(result_files, 'a') as f:
        #     # 写入coco指标、loss、learning rate
        #     result_info = [f'{i:.4f}' for i in coco_info + [mean_loss.item()]] + [f'{lr: .6f}']
        #     txt = 'epoch:{} {}'.format(epoch, '\t'.join(result_info))
        #     f.write(txt + '\n')
        # # pascal mAP
        # val_map.append(coco_info[1])
    torch.save(model.state_dict(), os.path.join(checkpoint_root, 'pretrain.pth'))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  second unfrozen backbone and train all network           #
    #  解冻前置特征提取网络权重（backbone），训练整个网络权重            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 冻结backbone部分底层权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split('.')[0]
        if split_name in ['0', '1', '2', '3']:
            parameter.requires_grid = False
        else:
            parameter.requires_grid = True
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    num_epochs = 20
    for epoch in range(init_epochs, num_epochs + init_epochs, 1):
        # mean_loss, lr = utils.train_one_epoch(
        #     model, optimizer, train_data_loader,
        #     device, epoch, print_freq=50, warmup=True, scaler=scaler
        # )
        # train_loss.append(mean_loss.item())
        # learning_rate.append(lr)
        # # 更新学习率
        # lr_scheduler.step()
        # coco_info = utils.evaluate(model, val_data_loader, device=device)
        # with open(result_files, 'a') as f:
        #     # 写入coco指标、loss、learning rate
        #     result_info = [f'{i:.4f}' for i in coco_info + [mean_loss.item()]] + [f'{lr: .6f}']
        #     txt = 'epoch:{} {}'.format(epoch, '\t'.join(result_info))
        #     f.write(txt + '\n')
        # # pascal mAP
        # val_map.append(coco_info[1])

        # 仅保留最后5个epoch的权重
        if epoch in range(num_epochs + init_epochs)[-5:]:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(save_files, os.path.join(checkpoint_root, 'mobile-model-epoch{}.pth'.format(epoch)))
        # TODO: plot loss and lr curve


if __name__ == '__main__':
    main('../')