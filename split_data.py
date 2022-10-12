import os
import random


def split_data(seed=3407, file_path='./VOCdevkit/VOC2012/Annotations', val_rate=0.5):
    random.seed(seed)
    assert os.path.exists(file_path), "path: '{}' does not exist.".format(file_path)
    file_name = sorted([file.split('.')[0] for file in os.listdir(file_path)])
    file_num = len(file_name)
    val_index = random.sample(range(0, file_num), k=int(file_num * val_rate))
    train_file = []
    val_file = []
    for index, name in enumerate(file_name):
        if index in val_index:
            val_file.append(name)
        else:
            train_file.append(name)
    train_f = open("train.txt", "w")
    eval_f = open("val.txt", "w")
    train_f.write('\n'.join(train_file))
    eval_f.write('\n'.join(val_file))


if __name__ == '__main__':
    split_data()
