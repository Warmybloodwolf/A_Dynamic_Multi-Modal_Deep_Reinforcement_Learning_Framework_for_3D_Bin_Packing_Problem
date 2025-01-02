#!/usr/bin/env python
# coding: utf-8

# In[269]:


import os
import pandas as pd
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, Dataset


# In[260]:


# h=(w_0*L+l_0*W)/(w_0+l_0)
# w = w_0/W l=l_0/L 
def scale_inst_(inst, bin_size):
    inst[:,2] /= (inst[:,0]*bin_size[1] + inst[:,1]*bin_size[0])/(inst[:,0]+inst[:,1])
    inst[:,0] /= bin_size[0]
    inst[:,1] /= bin_size[1]
    return inst


# In[261]:


def read_instance_(data_cycle):
    boxes = []
    instance_num = np.asarray(list(map(int, next(data_cycle))))
    bin_size = np.asarray(list(map(int, next(data_cycle))))
    box_num = int(next(data_cycle)[0])
    
    for _ in range(box_num):
        boxes.append(np.asarray(list(map(int, next(data_cycle)))))
        
    boxes = np.asarray(boxes)
    # print(boxes)
    
    box_sizes = boxes[:,[1,3,5]]
    box_num = boxes[:,-1]
    inst = np.repeat(box_sizes, box_num, axis=0)
    
    # we drop bin size!
    return inst, bin_size


# In[270]:


def read_ds(file):
    values = []
    instances = []
    bin_sizes = []
    
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            values.append(line.split())
            
    data_cycle = cycle(values)
    
    for inst in range(int(next(data_cycle)[0])):
        inst, bin_size = read_instance_(data_cycle)
        inst = inst.astype(float)
        bin_size = bin_size.astype(float)
        instances.append(inst)
        bin_sizes.append(bin_size)
        
    instances = [scale_inst_(inst, bin_size) for inst, bin_size in zip(instances, bin_sizes)]
    instances = np.vstack(instances)
        
    return instances


# In[275]:


def get_br_ds(path, graph_size=200, batch_size=32):
    
    insts1 = read_ds(os.path.join(path, "br1.txt"))
    insts2 = read_ds(os.path.join(path, "br2.txt"))
    insts3 = read_ds(os.path.join(path, "br3.txt"))
    insts4 = read_ds(os.path.join(path, "br4.txt"))
    insts5 = read_ds(os.path.join(path, "br5.txt"))
    insts6 = read_ds(os.path.join(path, "br6.txt"))

    insts8 = read_ds(os.path.join(path, "br8.txt"))
    insts9 = read_ds(os.path.join(path, "br9.txt"))

    training_ds = np.vstack([insts1,insts2,insts3,insts4,insts5,insts6,insts8,insts9])
    test_ds = read_ds(os.path.join(path, "br7.txt"))
    
    divide_size = graph_size*batch_size
    
    b_n = training_ds.shape[0]//divide_size
    training_ds = training_ds[0:divide_size*b_n , :]
    
    b_n = test_ds.shape[0]//graph_size
    test_ds = test_ds[0:graph_size*b_n , :]
    
    return training_ds, test_ds


def read_training_data(csv_file_path):
    """
    读取 CSV 文件并将数据组织为指定的格式。

    :param csv_file_path: CSV 文件路径
    :return: 数据集，格式为 LIST[(box_num, 3)]
    """
    # 读取 CSV 文件
    data = pd.read_csv(csv_file_path
                    #    , dtype={'value': 'float32'}
                       )

    # 按 instance_id 分组
    grouped_data = data.groupby('instance_id')

    dataset = []
    
    # 遍历每个分组并提取 (length, width, height)
    for _, group in grouped_data:
        # 提取 length, width, height 并转换为 (box_num, 3)
        box_data = group[['length', 'width', 'height']].to_numpy()
        # box_data *= 0.1
        # box_data = np.maximum(box_data, 1)
        dataset.append(box_data)

    return dataset


class BoxDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集。

        :param data: 数据集，格式为 LIST[(box_num, 3)]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def custom_collate_fn(batch):
    """
    自定义 collate_fn，确保每个 batch 的实例具有相同的 box_num。

    :param batch: 数据列表
    :return: 按 box_num 分组的 batch 数据
    """
    # 按 box_num 分组
    grouped_batches = {}
    for instance in batch:
        box_num = instance.shape[0]
        if box_num not in grouped_batches:
            grouped_batches[box_num] = []
        grouped_batches[box_num].append(instance)

    final_batches = []
    for box_num, instances in grouped_batches.items():
        # 如果超过128，按最大 batch_size=128 划分
        for i in range(0, len(instances), 128):
            final_batches.append(np.stack(instances[i:i + 128]))

    return final_batches


