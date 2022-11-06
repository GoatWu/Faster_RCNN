# 训练工具集简介

## 数据集简介

### PASCAL VOC2012

#### 数据集简介

**任务简介：**

- 分割（Segmentation）：实力分割、语义分割
- 检测（Detection）
- 分类（Multi-Label Classification）

**类别简介：**

一共 $20$ 个类，分为交通工具、家庭用具、动物、人这四个大类

**文件结构：**

```
|---- Annotatioins             # 所有图片的标注信息（一张图片对应一个）
|---- ImageSets
|    |----Action               # 人的行为动作图像信息
|    |----Layout               # 人的各个部位图像信息
|    |----Main                 # 目标检测分类图像信息
|    |    |----train.txt       # 训练集 5717
|    |    |----val.txt         # 验证集 5823
|    |    |----trainval.txt    # 训练集+验证集 11540
|    |---- Segmentation        # 目标分割图像信息
|---- JPEGImages               # 所有图像文件
|---- SegmentationClass        # 图像分割png图（基于类别）
|---- SegmentationObject       # 图像分割png图（基于目标）
```

**Main文件夹：目标检测主文件夹**

- `train.txt`：
  ```
  2008_000008      # 每一行为一张图片的名称，构成训练集
  2008_000015
  ......
  ```
- `val.txt`：同理
- `boat_train`：训练集中有关于 `boat` 类别的图像信息
  ```
  2008_000189 -1       # -1表示图片中不存在boat目标
  2008_000191  0       #  0表示图片中的boat目标比较难以检测
  2008_000148  1       #  1表示图片中有boat目标
  ```

#### 构建自己的VOC格式数据集

**数据标注工具**

[https://github.com/heartexlabs/labelImg](https://github.com/heartexlabs/labelImg)：生成 `.xml` 格式的标注工具

1. 构建 `classes.txt`：
   ```
   dog
   person
   cat
   tv
   ...
   ```
2. 准备好图片文件夹：`image`
3. 创建一个空文件夹：`annotations`
4. 在当前文件夹输入命令，进入标注图片GUI：
   `labelImg ./image ./classes,txt`
5. 界面中点击 `Change Save Dir` 选择标注文件的保存路径；
6. 点击 `Create RectBox` 开始绘制标注框，会自动跳出一个框来选择类别；
   可以勾选 difficult 来选择是否难以检测
7. 点击 `save` 即可保存标注

**生成 train.txt 和 val.txt**

`train_utils/split_data.py` 脚本可以将 `train_val.txt` 分割为 `train.txt` 和 `val.txt`，即遍历 `Annotations` 目录然后进行随机采样。


### Microsoft COCO

#### 数据集简介

**数据集主要特性**

- Object Segmentation：目标级分割
- Recognition in context：图像情景识别
- Superpixel stuff segmentation：超像素分割
- 330K images (>200K labeled)：超过33万张图片，超过20万张图片被标注过
- 1.5 million object instance：150万个对象实例
- 80 object categories：80个目标类别
- 91 stuff categories：91个材料类别
- 5 captions per image：每张图像有5段情景描述
- 250000 people with keypoint：对25个人进行了关键点标注

**stuff 和 object 的区别**

- stuff：没有明确边界的材料和对象（例如：天空）。
- object：stuff中91类的一个子集。在目标检测中，一般只使用object80类即可。

**文件结构**

```
├── coco2017: 数据集根目录
     ├── train2017: 所有训练图像文件夹(118287张)
     ├── val2017: 所有验证图像文件夹(5000张)
     └── annotations: 对应标注文件夹
     		  ├── instances_train2017.json: 对应目标检测、分割任务的训练集标注文件
     		  ├── instances_val2017.json: 对应目标检测、分割任务的验证集标注文件
     		  ├── captions_train2017.json: 对应图像描述的训练集标注文件
     		  ├── captions_val2017.json: 对应图像描述的验证集标注文件
     		  ├── person_keypoints_train2017.json: 对应人体关键点检测的训练集标注文件
     		  └── person_keypoints_val2017.json: 对应人体关键点检测的验证集标注文件夹
```

- 训练集中的一些图片有标注的问题（目标边界框大小为0），需要筛选掉
- 基本只需要用到 `instances_train2017.json` 和 `instances_val2017.json`

**instances_val2017.json 结构**

- `'images' = {list: 5000}`：每张图片的名称、高宽等信息
- `‘annotations’ = {list: 36781}`：每个目标的标注，标注其 所属图片标号、bbox 信息、类别标号等
- `‘categories’ = {list: 80}`：80个object类别，有超类名、小类名、类别标号三个项

#### pycocotools的使用

```python
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib,pyplot as plt


json_path = './instances_val2017.json'
img_path = './val2017'

# 载入数据
coco = COCO(annotation_file=json_path)
# 获取每张图片的索引
ids = list(sorted(coco.imgs.keys()))
# 获取所有类别（91类）的标号以及名称
coco_classes = dict([(v['id'], v['name']) for k, v in coco.cats.items()])
# 遍历前三张图片
for img_id in ids[:3]:
    # 获取对应图像id的所有annotations idx信息
    ann_ids = coco.getAnnIds(imgIds=img_id)
    # 根据annotations idx信息获取所有标注信息
    targets = coco.loadAnns(ann_ids)
    # 获取图片名称	path: '000000000139.jpg'
    path = coco.loadImgs(img_id)[0]['file_name']
    # 读取图片
    img = Image.open(os.path.join(img_path, path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    # 在图片上绘制框
    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x + w), int(x + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target['category_id']])
    plt.imshow(img)
    plt.show()
    
```

#### mAP计算

在验证集上得到的预测结果使用一个列表保存：

```python
[{
    'image_id'		: int,
    'category_id'	: int,
    'bbox'			: [x,y,width,height],
    'score'			: float,
}]
```

- 分别表示：图像标号、在stuff91上的类别标号、bbox信息、预测分数；

- 保存到 `predict_results.json` 中。
- 使用下面的命令对比预测文件和真实标签，得到各个指标

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# 载入标注文件
coco_true = COCO(annotation_file='./annotations')
# 载入预测结果
coco_pre = coco_true.loadRes('predict_results.json')
coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType='bbox')
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()

```

### COCO评价标准

**目标检测中的AP（Average Precision）**

- 首先得到每个预测框的置信概率（confidence）和每个预测框是否匹配到真实框（例如设置阈值为 $0.5$），同时记录这种类别的总共真实框数目。

- 将所有预测框按照 confidence 从大到小进行排序，得到表格（假设真实框数目为 $5$）：

  | GT ID | Confidence | OB(IOU=0.5) |
  | :---: | :--------: | :---------: |
  |   1   |    0.98    |    True     |
  |   3   |    0.89    |    True     |
  |   7   |    0.78    |    True     |
  |   3   |    0.66    |    False    |
  |   4   |    0.52    |    True     |

- 从上到下取每个 confidence 为阈值，得到表格：

  | Rank | Precition | Recall |
  | :--: | :-------: | :----: |
  |  1   |    1.0    |  0.2   |
  |  2   |    1.0    |  0.4   |
  |  3   |    1.0    |  0.6   |
  |  4   |   0.75    |  0.6   |
  |  5   |    0.8    |  0.8   |

- 相同地 Recall 只取 Precision 的最大值:

  | Rank | Precition | Recall |
  | :--: | :-------: | :----: |
  |  1   |    1.0    |  0.2   |
  |  2   |    1.0    |  0.4   |
  |  3   |    1.0    |  0.6   |
  |  5   |    0.8    |  0.8   |

- 阴影部分的面积即为 AP
- 所有类的 AP 的平均值即为 mAP

**COCO 中的指标**

> - Average Precision（AP）:
>   - AP：当 `IoU=(0.50, 0.55, 0.55, ..., 0.95)` 时的10个AP的平均值
>   - AP（IOU=0.5）：当IoU大于等于0.5时视为预测正确
>   - AP（IOU=0.75）：当IoU大于等于0.75时视为预测正确
> - AP Across Scales：
>   - AP-small：$\text{area} < 32^2$
>   - AP-medium：$32^2 < \text{area} < 96^2$
>   - AP-small：$\text{area} > 96^2$
> - Average Recall（AR）：
>   - AR（max=1）：每张图片最多提供 $1$ 个预测框的预测结果
>   - AR（max=10）：每张图片最多提供 $10$ 个预测框的预测结果
>   - AR（max=100）：每张图片最多提供 $100$ 个预测框的预测结果
> - AR Across Scales：
>   - AR-small：$\text{area} < 32^2$
>   - AR-medium：$32^2 < \text{area} < 96^2$
>   - AR-small：$\text{area} > 96^2$

以上指标中：

- AP 为 COCO 的主要指标，需要特别注意；
- AP（IOU=0.5）为 Pascal VOC 的指标，也需要注意。
- 其他指标根据不同的感兴趣目标而定。

## group_by_aspect_ratio.py

### pytorch官方的BatchSampler实现

```python
class BatchSampler(Sampler[List[int]]):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

- 基于一个可迭代的Sampler，如 `torch.utils.data.RandomSampler`
- `for idx in self.sampler` 每次取出一个随机元素，并放入 batch 中，满则返回此 batch

### 这里的GroupBatchSampler实现

- 目的：按图片相似高宽比采样图片组成batch，减小训练时所需的GPU显存

#### GroupBatchSampler思路

- 将所有的图像进行分组，每个图像有一个组标签，相同组标签的图片构成一个batch
- `for idx in self.sampler` 每次取出一个随机元素，放入一个类似于 `map<int, vector<int>>` 的容器里面。`vector<int>` 满了之后则取出，返回这个batch
- 对于每个group中剩下的元素，从已经被选择过的同组元素中采样补满，组成一个新的batch

#### 划分组别 group_id

按照高宽比从 $0.5-2.0$ 划分出 $2k+1$ 个点，加上两端（即小于 $0.5$ 或大于 $2.0$ ）共 $2k+2$ 个区间。计算得到每张图片的高宽比，并分入这些组中，得到 `group_id`。
