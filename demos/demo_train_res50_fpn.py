import os
import datetime
import torch
import torch.utils.data
import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups


def create_model(num_classes, load_pretrain_weights=True):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    backbone = resnet50_fpn_backbone(pretrain_path="../weights/resnet50.pth",
                                     trainable_layers=3)
    model = FasterRCNN(backbone=backbone, num_classes=91)   # pretrained num_classes
    if load_pretrain_weights:
        weights_dict = torch.load('../weights/fasterrcnn_resnet50_fpn_coco.pth', map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    # 改为自己数据集的num_classes
    in_features = model.roi_heads.box_predictor.cls_loss.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using {} device training.'.format(device.type))
    root, VOC_root = args.root_path, args.root_path
    output_dir = args.output_dir
    results_files = 'results{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    checkpoint_root = os.path.join(root, output_dir, 'ResNet50_FPN')
    result_files = os.path.join(checkpoint_root, results_files)
    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root)
    data_transform = {
        'train': transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        'val': transforms.Compose([transforms.ToTensor()])
    }
    if os.path.exists(os.path.join(VOC_root, 'VOCdevkit')) is False:
        raise FileNotFoundError('VOCdevkit does not exist in path: "{}".'.format(VOC_root))
    train_dataset = VOCDataSet(voc_root=VOC_root, year='2012', transforms=data_transform['train'], txt_name='train.txt')
    train_sampler, train_batch_sampler = None, None
    # 是否按照图片高宽比采样图片组成batch
    # 是的话能减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    batch_size = args.batch_size
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
    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_dacay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)
    if args.resume != '':
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        print('the training process resumes from epoch{}...'.format(args.start_epoch))

    train_loss, learning_rate, val_map = [], [], []

    for epoch in range(args.start_epoch, args.epochs):
        # TODO: utils.train_one_epoch
        # mean_loss, lr = utils.train_one_epoch(
        #     model, optimizer, train_data_loader,
        #     device, epoch, print_freq=50, warmup=True, scaler=scaler
        # )
        # train_loss.append(mean_loss.item())
        # learning_rate.append(lr)
        # lr_scheduler.step()
        # coco_info = utils.evaluate(model, val_data_loader, device=device)
        # with open(result_files, 'a') as f:
        #     # 写入coco指标、loss、learning rate
        #     result_info = [f'{i:.4f}' for i in coco_info + [mean_loss.item()]] + [f'{lr: .6f}']
        #     txt = 'epoch:{} {}'.format(epoch, '\t'.join(result_info))
        #     f.write(txt + '\n')
        # # pascal mAP
        # val_map.append(coco_info[1])
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if args.amp:
            save_files['scaler'] = scaler.state_dict()
        torch.save(save_files, os.path.join(checkpoint_root, 'ResnetFPN-model-epoch{}.pth'.format(epoch)))
        # TODO: plot loss and lr curve


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--root_path', default='../', help='root path of the project.')
    parser.add_argument('--num_classes', default=20, help='num_classes without background.')
    parser.add_argument('output_dir', default='save_checkpoints', help='checkpoints folder.')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch, use for resume training.')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    main(args)
