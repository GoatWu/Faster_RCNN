import time
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):

    def __init__(self, voc_root, year='2012', transforms=None, txt_name: str = "train.txt"):
        assert year in ['2007', '2012']
        self.root = os.path.join(voc_root, 'VOCdevkit', f"VOC{year}")
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        print('**********************')
        self.display_time()
        print('INFO: start making VOC Dataset.')
        txt_path = os.path.join(self.root, 'ImageSets', 'Main', txt_name)
        with open(txt_path, 'r') as f:
            xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                        for line in f.readlines() if len(line.strip()) > 0]
        self.xml_list = []
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f'Warning: 404 no found "{xml_path}", skip this annotation file.')
                continue
            with open(xml_path) as f:
                xml_str = f.read()
            xml = etree.fromstring(xml_str)
            xml_data = self.parse_xml_to_dict(xml)['annotation']
            if 'object' not in xml_data:
                print(f'Warning: no object in "{xml_path}", skip this annotation filw.')
                continue
            self.xml_list.append(xml_path)
        self.display_time()
        print("INFO: Finish making VOC Dataset.")
        print('**********************')

        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "json dict file not exist"
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        """ 传入一个索引值，返回索引值对应的图片和标签 """
        xml_path = self.xml_list[index]
        with open(xml_path) as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        xml_data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, xml_data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError("Image '{}' format not JPEG".format(img_path))
        # bounding_box, 标签, 是否重叠
        boxes, labels, iscrowd = [], [], []
        for obj in xml_data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            # iscrowd这里表示分类难度，0: 简单, 1: 困难
            iscrowd.append(int(obj['difficult']) if 'difficult' in obj else 0)
        # convert everything into torch.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1)
        target = {'boxes': boxes, 'labels': labels, 'iscrowd': iscrowd,
                  'area': area, 'image_id': image_id}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, index):
        xml_path = self.xml_list[index]
        with open(xml_path, 'r') as f:
            xml_str = f.read()
        xml = etree.fromstring(xml_str)
        xml_data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(xml_data['size']['height'])
        data_width = int(xml_data['size']['height'])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件以字典形式存储
        注意可能有多个object对象，因此使用列表存储各个object
        """
        if len(xml) == 0:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    @staticmethod
    def display_time():
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), end='\t\t')


def test_dataset():
    import transforms
    from draw_box_utils import draw_objs
    import matplotlib.pyplot as plt
    import torchvision.transforms as ts
    import random
    category_index = {}
    try:
        json_file = open('./pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        category_index = {str(v): str(k) for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # load train data set
    train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
    print(len(train_data_set))
    for index in random.sample(range(0, len(train_data_set)), k=5):
        img, target = train_data_set[index]
        img = ts.ToPILImage()(img)
        plot_img = draw_objs(img,
                             target["boxes"].numpy(),
                             target["labels"].numpy(),
                             np.ones(target["labels"].shape[0]),
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.savefig('./test_images/test{}.png'.format(index))


if __name__ == '__main__':
    test_dataset()
