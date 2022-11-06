import bisect
from collections import defaultdict
import copy
from itertools import repeat, chain
import math
import numpy as np

import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm
import torchvision
from PIL import Image


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError('sampler should be an instance of '
                             'torch.utils.data.Sampler, but got sampler={}'.format(sampler))
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)    # 当 key 不存在时，返回默认的 list，也即空列表
        samples_per_group = defaultdict(list)
        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size
        expect_num_batches = len(self)
        num_remaining = expect_num_batches - num_batches
        if num_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                remaining = self.batch_size - len(buffer_per_group[group_id])
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    print("Warning: Your dataset doesn't support the fast path"
          "for computing the aspect ratios, so will iterate over "
          "the full dataset and load every image instead. "
          "This might take some time...")
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = Sampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14, collate_fn=lambda x:x[0]
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratio_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratio_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info['width']) / float(img_info['height'])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratio_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratio_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset, indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, 'get_height_and_width'):
        return _compute_aspect_ratio_custom_dataset(dataset, indices)
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratio_coco_dataset(dataset, indices)
    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratio_voc_dataset(dataset, indices)
    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratio_subset_dataset(dataset, indices)
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    # bisect_right: 寻找y元素应该排在bins中哪个元素的右边，返回索引
    # map(lamda y: y ** 2, [1, 2, 3]) = [1, 4, 9]
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    # 将 [0.5, 2] 区间划分为 2k 份（2k+1 个点，2k个区间）
    # 这些点去对数后为等差数列
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    # 返回 group_ratios 中每个值在 bins 中应处于的位置
    groups = _quantize(aspect_ratios, bins)
    # 统计每个区间的频数
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print('Using {} as bins for aspect ratio quantization.'.format(fbins))
    print('Count of instances per bin: {}.'.format(counts))
    return groups


if __name__ == '__main__':
    k = 3
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    print(bins)
    aspect_ratios = [0.3, 0.5, 0.6, 0.66, 0.8, 1.1, 1.2, 1.5, 1.7, 1.98, 2.0, 2.2]
    groups = _quantize(aspect_ratios, bins)
    for (aspect_ratio, group) in zip(aspect_ratios, groups):
        print('{}: group {}'.format(aspect_ratio, group))
