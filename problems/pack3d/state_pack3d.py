#!/usr/bin/env python
# coding: utf-8


import torch
from typing import NamedTuple
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.functions import move_to

class PackAction():
    # (batch, 1)

    def __init__(self, batch_size, device):
        self.index = torch.zeros(batch_size, 1, device=device)
        self.x = torch.empty(
            batch_size, 1, device=device).fill_(-2)  # set to -2
        self.y = torch.empty(batch_size, 1, device=device).fill_(-2)
        self.z = torch.empty(batch_size, 1, device=device).fill_(-2)
        self.rotate = torch.zeros(batch_size, 1,device=device)
        self.updated_shape = torch.empty(batch_size, 3, device=device)
        self.sp = torch.FloatTensor(batch_size,  1).fill_(1)
        self.sp=move_to(self.sp,device)

        '''
        0: no rotate
        1: (x,y,z) -> (y,x,z)
        2: (x,y,z) -> (y,z,x)
        3: (x,y,z) -> (z,y,x)
        4: (x,y,z) -> (z,x,y)
        5: (x,y,z) -> (x,z,y)
        '''

    def set_index(self, selected):
        self.index = selected

    def set_rotate(self, rotate):
        self.rotate = rotate

    def set_shape(self, length, width, height):
        # (batch, 3)
        self.updated_shape = torch.stack([length, width, height], dim=-1)

    def get_shape(self):
        return self.updated_shape

    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_packed(self):
        return torch.cat((self.sp,self.updated_shape, self.x, self.y, self.z), dim=-1)

    def reset(self):
        self.__init__(self.index.size(0))

    def __call__(self):
        return {'index': self.index,
                'rotate': self.rotate,
                'x': self.x,
                'y': self.y}

    def __len__(self):
        return self.index.size(0)


def push_to_tensor_alternative(tensor, x):
    return torch.cat((tensor[:, 1:, :], x), dim=1)


class StatePack3D():

    def __init__(self, batch_size, block_size, device, position_size=128):
        self.device = device
        self.batch_size=batch_size
        # (batch, block_size, 3)
        self.raw_data = None
        self.scale = torch.zeros(
            batch_size, 10, 3, dtype=torch.float, device=device)
        self.container_index = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        # 当前container以外的体积
        self.container_volume = torch.zeros(batch_size, 1, device=device)
        self.valid_volume = torch.zeros(batch_size, 1, device=device)

        # {length| width| height| x| y| z}
        self.packed_state = torch.zeros(
            batch_size, block_size, 7, dtype=torch.float, device=device)
        self.packed_rotate = torch.zeros(
            batch_size, block_size, 1, dtype=torch.int64, device=device)
        self.total_rewards = torch.zeros(
            batch_size, dtype=torch.float, device=device)
        self.action = PackAction(batch_size, device=device)
        self.container_size = [10, 10, 10]
        self.blocks_num = block_size
        self.block_dim = 3

        self.container = np.zeros([batch_size,10,10,10]).astype(int)
        self.positions = np.zeros((batch_size,self.blocks_num, self.block_dim)).astype(int)
        self.reward_type = "C-soft"
        self.stable = np.zeros((batch_size,self.blocks_num),dtype=bool)
        self.valid_size = [0]*batch_size
        self.empty_size = [0]*batch_size
        self.heightmap = np.zeros([batch_size,10,10]).astype('int')
        self.index=0

    def put_reward(self, reward):
        self.total_rewards += reward

    def get_rewards(self):
        return self.total_rewards

    def update_env(self, batch,batch_size,block_size,device):
        sp_initial=torch.FloatTensor(batch_size,block_size,1).fill_(0)
        sp_initial=move_to(sp_initial,device)
        position_initial=torch.FloatTensor(batch_size,block_size,3).fill_(0)
        position_initial=move_to(position_initial,device)
        self.packed_state=torch.cat([sp_initial,batch,position_initial],dim=2)
    
    def init_env(self, batch,batch_size,block_size,device):
        self.raw_data = batch
        # TODO: 根据每个instance的box数据决定第一个container的尺寸，这里batch为(batch_size,block_size,3)的numpy数组，记录长宽高数据
        # 假设决策得到选用的container尺寸为35，23，13和54, 45, 36
        container_size = np.zeros((batch_size, 3))
        container_size[0] = [35,23,13]
        container_size[1:] = [54,45,36]
        embedding_size = np.array([10, 10, 10])
        
        embedding_expand = np.expand_dims(embedding_size, 0)
        # B, 3
        scale = embedding_expand / container_size
        self.scale[:, 0, :] = torch.tensor(scale)
        scale_expand = torch.tensor(np.expand_dims(scale, 1))
        scale_expand = move_to(scale_expand, device)
        resized_data = batch * scale_expand
        resized_data[resized_data < 1] = 1.0
        resized_data[resized_data > 10] = 10.0
        self.update_env(resized_data,batch_size,block_size,device)
        
    def update_select(self, selected):
        self.action.set_index(selected)
        box_length, box_width, box_height = self._get_action_box_shape()

        self.action.set_shape(box_length, box_width, box_height)

    def _get_action_box_shape(self):
        select_index = self.action.index.long()

        box_raw_l = self.packed_state[:, :, 1].squeeze(-1)
        box_raw_w = self.packed_state[:, :, 2].squeeze(-1)
        box_raw_h = self.packed_state[:, :, 3].squeeze(-1)

        box_length = torch.gather(box_raw_l, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_w, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_h, -1, select_index).squeeze(-1)

        return box_length, box_width, box_height

    def update_rotate(self, rotate):

        self.action.set_rotate(rotate)

        # there are 5 rotations except the original one
        rotate_types = 5
        batch_size = rotate.size()[0]

        rotate_mask = torch.empty((rotate_types, batch_size), dtype=torch.bool)
        rotate_mask = move_to(rotate_mask, 'cuda')
        select_index = self.action.index.long()

        box_raw_x = self.raw_data[:, :, 0].squeeze(-1)
        box_raw_y = self.raw_data[:, :, 1].squeeze(-1)
        box_raw_z = self.raw_data[:, :, 2].squeeze(-1)

        # (batch)  get the original box shape
        box_length = torch.gather(box_raw_x, -1, select_index).squeeze(-1)
        box_width = torch.gather(box_raw_y, -1, select_index).squeeze(-1)
        box_height = torch.gather(box_raw_z, -1, select_index).squeeze(-1)

        scale_x = self.scale[:, :, 0].squeeze(-1)
        scale_y = self.scale[:, :, 1].squeeze(-1)
        scale_z = self.scale[:, :, 2].squeeze(-1)

        select_index = self.container_index.long()

        selected_scale_x = torch.gather(scale_x, -1, select_index).squeeze(-1)
        selected_scale_y = torch.gather(scale_y, -1, select_index).squeeze(-1)
        selected_scale_z = torch.gather(scale_z, -1, select_index).squeeze(-1)

        for i in range(rotate_types):
            rotate_mask[i] = rotate.squeeze(-1).eq(i + 1)

        # rotate in 5 directions one by one
        # (x,y,z)->(y,x,z)
        # (x,y,z)->(y,z,x)
        # (x,y,z)->(z,y,x)
        # (x,y,z)->(z,x,y)
        # (x,y,z)->(x,z,y)
        inbox_length = box_length * selected_scale_x
        inbox_width = box_width * selected_scale_y
        inbox_height = box_height * selected_scale_z
        for i in range(rotate_types):

            if i == 0:
                box_l_rotate = box_width * selected_scale_y
                box_w_rotate = box_length * selected_scale_x
                box_h_rotate = box_height * selected_scale_z
            elif i == 1:
                box_l_rotate = box_width * selected_scale_y
                box_w_rotate = box_height * selected_scale_z
                box_h_rotate = box_length * selected_scale_x
            elif i == 2:
                box_l_rotate = box_height * selected_scale_z
                box_w_rotate = box_width * selected_scale_y
                box_h_rotate = box_length * selected_scale_x
            elif i == 3:
                box_l_rotate = box_height * selected_scale_z
                box_w_rotate = box_length * selected_scale_x
                box_h_rotate = box_width * selected_scale_y
            elif i == 4:
                box_l_rotate = box_length * selected_scale_x
                box_w_rotate = box_height * selected_scale_z
                box_h_rotate = box_width * selected_scale_y

            box_l_rotate = torch.masked_select(
                box_l_rotate, rotate_mask[i])
            box_w_rotate = torch.masked_select(
                box_w_rotate, rotate_mask[i])
            box_h_rotate = torch.masked_select(
                box_h_rotate, rotate_mask[i])

            inbox_length = inbox_length.masked_scatter(
                rotate_mask[i], box_l_rotate)
            inbox_width = inbox_width.masked_scatter(
                rotate_mask[i], box_w_rotate)
            inbox_height = inbox_height.masked_scatter(
                rotate_mask[i], box_h_rotate)
            
        inbox_length[inbox_length < 1] = 1.0
        inbox_length[inbox_length > 10] = 10.0
        inbox_width[inbox_width < 1] = 1.0
        inbox_width[inbox_width > 10] = 10.0
        inbox_height[inbox_height < 1] = 1.0
        inbox_height[inbox_height > 10] = 10.0
        
        
        self.packed_rotate[torch.arange(0, rotate.size(
            0)), select_index.squeeze(-1), 0] = rotate.squeeze(-1)

        self.action.set_shape(inbox_length, inbox_width, inbox_height)

    def update_pack(self):
        # batch_size = self.boxes.size(0)
        select_index = self.action.index.squeeze(-1).long().tolist()

        # z = self._get_z_skyline(x, y)
        x=torch.tensor(self.positions[:,self.index,0]).unsqueeze(-1).float()
        y=torch.tensor(self.positions[:,self.index,1]).unsqueeze(-1).float()
        z=torch.tensor(self.positions[:,self.index,2]).unsqueeze(-1).float()
        x=move_to(x,self.device)
        y=move_to(y,self.device)
        z=move_to(z,self.device)
        self.action.set_pos(x, y, z)

        packed_box = self.action.get_packed()

        mid=self.packed_state.clone()
        self.packed_state=mid
        for i in range(self.batch_size):
            self.packed_state[i][select_index[i]]=packed_box[i]

        self.index += 1
        if self.index >= self.blocks_num:
            return True     # pack done
        else:
            return False

    def _get_z(self, x, y):

        inbox_length = self.action.get_packed()[:, 0]
        inbox_width = self.action.get_packed()[:, 1]

        in_back = torch.min(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_front = torch.max(x.squeeze(-1), x.squeeze(-1) + inbox_length)
        in_left = torch.min(y.squeeze(-1), y.squeeze(-1) + inbox_width)
        in_right = torch.max(y.squeeze(-1), y.squeeze(-1) + inbox_width)

        box_length = self.packed_cat[:, :, 0]
        box_width = self.packed_cat[:, :, 1]
        box_height = self.packed_cat[:, :, 2]

        box_x = self.packed_cat[:, :, 3]
        box_y = self.packed_cat[:, :, 4]
        box_z = self.packed_cat[:, :, 5]

        box_back = torch.min(box_x, box_x + box_length)
        box_front = torch.max(box_x, box_x + box_length)
        box_left = torch.min(box_y, box_y + box_width)
        box_right = torch.max(box_y, box_y + box_width)
        box_top = torch.max(box_z, box_z + box_height)

        in_back = in_back.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_front = in_front.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])
        in_left = in_left.unsqueeze(-1).repeat([1, self.packed_cat.size()[1]])
        in_right = in_right.unsqueeze(-1).repeat(
            [1, self.packed_cat.size()[1]])

        is_back = torch.gt(box_front, in_back)
        is_front = torch.lt(box_back, in_front)
        is_left = torch.gt(box_right, in_left)
        is_right = torch.lt(box_left, in_right)

        is_overlaped = is_back * is_front * is_left * is_right
        non_overlaped = ~is_overlaped

        overlap_box_top = box_top.masked_fill(non_overlaped, 0)

        max_z, _ = torch.max(overlap_box_top, -1, keepdim=True)

        return max_z

    def get_boundx(self):
        batch_size = self.packed_state.size()[0]
        front_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 0]

        return front_bound

    def get_boundy(self):

        batch_size = self.packed_state.size()[0]
        right_bound = torch.ones(
            batch_size, device=self.packed_state.device) - self.action.get_shape()[:, 1]

        return right_bound

    def get_height(self):
        return np.max(self.heightmap,axis=(1,2))

    def get_gap_size(self):

        bin_volumn = torch.tensor(self.get_height()) * 100.0 
        bin_volumn = move_to(bin_volumn,self.device)

        # print(f'bin_volumn:{bin_volumn}')
        # print(f'container_index:{self.container_index}')

        x_scale = torch.gather(self.scale[:, :, 0], -1, self.container_index).squeeze(-1)
        y_scale = torch.gather(self.scale[:, :, 1], -1, self.container_index).squeeze(-1)
        z_scale = torch.gather(self.scale[:, :, 2], -1, self.container_index).squeeze(-1)

        valid_size = move_to(torch.tensor(self.valid_size),self.device)

        # print(f'valid_size:{valid_size}')
        # print(f'x_scale:{x_scale}')
        # print(f'y_scale:{y_scale}')
        # print(f'z_scale:{z_scale}')

        gap_volumn = (bin_volumn - valid_size) / (x_scale * y_scale * z_scale) 
        gap_volumn = gap_volumn + self.container_volume.squeeze(-1) - self.valid_volume.squeeze(-1)
        
        return gap_volumn

    def get_gap_ratio(self):

        bin_volumn = torch.tensor(self.get_height()) * 100.0 
        bin_volumn = move_to(bin_volumn,self.device)

        x_scale = torch.gather(self.scale[:, :, 0], -1, self.container_index).squeeze(-1)
        y_scale = torch.gather(self.scale[:, :, 1], -1, self.container_index).squeeze(-1)
        z_scale = torch.gather(self.scale[:, :, 2], -1, self.container_index).squeeze(-1)

        bin_volumn = bin_volumn / (x_scale * y_scale * z_scale) + self.container_volume.squeeze(-1)
        ebselong=torch.tensor([0.001])
        ebselong=move_to(ebselong,self.device)

        gap_ratio = self.get_gap_size() / (bin_volumn+ebselong)

        return gap_ratio

    def get_graph(self):
        return self.packed_cat
    
    def new_container(self,batch_index):
        # 第i个instance需要创建新的container
        embedding_size = np.array([10, 10, 10])
        self.container_volume[batch_index] += embedding_size[0] * embedding_size[1] * embedding_size[2] \
                      / self.scale[batch_index, self.container_index[batch_index], 0] \
                      / self.scale[batch_index, self.container_index[batch_index], 1] \
                      / self.scale[batch_index, self.container_index[batch_index], 2]
        self.valid_volume[batch_index] += self.valid_size[batch_index] \
                      / self.scale[batch_index, self.container_index[batch_index], 0] \
                      / self.scale[batch_index, self.container_index[batch_index], 1] \
                      / self.scale[batch_index, self.container_index[batch_index], 2]
        self.container_index[batch_index] += 1
        # TODO: 这里需要根据instance的box数据，创建新的container，并更新状态
        # 假设创建的container尺寸为42，30，40
        container_size = np.array([42, 30, 40])
        scale = embedding_size / container_size
        scale = move_to(torch.tensor(scale, dtype=torch.float32),self.device)
        self.scale[batch_index, self.container_index[batch_index], :] = scale
        scale_expand = scale.unsqueeze(0)
        # block, 3
        resized_data = self.raw_data[batch_index] * scale_expand
        resized_data = move_to(torch.tensor(resized_data),self.device)
        resized_data[resized_data < 1] = 1.0
        resized_data[resized_data > 10] = 10.0

        self.packed_state[batch_index, :, 1:4] = resized_data
        self.container[batch_index] = 0
        self.heightmap[batch_index] = 0
        self.valid_size[batch_index] = 0
        self.empty_size[batch_index] = 0 
        
        

