import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
import cv2
import ast
from tqdm import  tqdm

import glob
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class Resnet_FPN_merge(nn.Module):
    def __init__(self,out_channels = 256):
        super(Resnet_FPN_merge, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.to(device)
        self.layer_1 = nn.Sequential(*list(self.resnet.children())[:4])
        self.layer_2 = self.resnet.layer2 
        self.layer_3 = self.resnet.layer3
        self.layer_4 = self.resnet.layer4

        self.lateral4 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        self.output4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.output1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        if (torch.isnan(x).any()):
            print("error at Resnet_FPN")
        c1 = self.layer_1(x)
        if (torch.any(torch.isnan(c1))):
            print('error resnet ')
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')

        p4 = self.output4(p4)
        p3 = self.output3(p3)
        p2 = self.output2(p2)
        p1 = self.output1(p1)

        return p1, p2, p3, p4


def convert_rois_to_boxes(rois):
    MxN = (torch.max(rois[:,0])+1)*(torch.max(rois[:,1])+1).item()
    MxN = int(MxN)
    batch_index = torch.zeros(MxN)
    x1 = rois[:,2]
    y1 = rois[:,3]
    x2 = rois[:,4]
    y2 = rois[:,5]

    boxes = torch.stack([batch_index,x1,y1,x2,y2],dim=1)
    return boxes

def roi_align(feature_map, rois, output_size = (7,7)):
    return ops.roi_align(feature_map, rois, output_size , spatial_scale=1/4)

class convert_rois_to_box(nn.Module):
    def __init__(self):
        super(convert_rois_to_box, self).__init__()
        # pass
    def forward(self, rois):
        MxN = (torch.max(rois[:,0])+1)*(torch.max(rois[:,1])+1).item()
        MxN = int(MxN)
        batch_index = torch.zeros(MxN)

        batch_index = batch_index.to(device)
        x1 = rois[:,2]
        y1 = rois[:,3]
        x2 = rois[:,4]
        y2 = rois[:,5]

        boxes = torch.stack([batch_index,x1,y1,x2,y2],dim=1)
        return boxes

class position_embedding(nn.Module):
    def __init__(self, num_paches = 100, projection_dims = 256):
        super(position_embedding, self).__init__()
        self.x_position_embeddings = nn.Embedding(num_paches, projection_dims)
        nn.init.constant_(self.x_position_embeddings.weight, 0.)
        self.y_position_embeddings = nn.Embedding(num_paches, projection_dims)
        nn.init.constant_(self.y_position_embeddings.weight, 0.)
    def forward(self, rois):
        x = torch.max(rois[:,1])
        y = torch.max(rois[:,0])

        x = x.to(torch.int32)
        y = y.to(torch.int32)
        x_1 = x+1
        y_1 = y+1
        
        col = torch.arange(x+1)
        col = col.to(device)
        row = torch.arange(y+1)
        row = row.to(device)

        x_pos = col.repeat(y_1)
        y_pos = row.repeat_interleave(x_1)

        x_embed = self.x_position_embeddings(x_pos)
        y_embed = self.y_position_embeddings(y_pos)
        
        return x_embed, y_embed

class predict_head_row(nn.Module):
    def __init__(self,hidden_dim = 512):
        super(predict_head_row, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff1 = nn.Linear(in_features=512, out_features=512)
        self.ff2 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.ff1(x)
        x = self.relu(x)
        x= self.ff2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x
class predict_head_col(nn.Module):
    def __init__(self,hidden_dim = 512):
        super(predict_head_col, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff1 = nn.Linear(in_features=512, out_features=512)
        self.ff2 = nn.Linear(in_features=512, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.ff1(x)
        x = self.relu(x)
        x= self.ff2(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim = 512, num_layers = 3, num_heads = 8, ff_dim = 512):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_dim = input_dim

    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        output = self.transformer_encoder(src)
        return output

class Model_merge(nn.Module):
    def __init__(self, num_layers = 3 ):
        super(Model_merge, self).__init__()
        self.pos_embedding = position_embedding()
        self.convert_rois_to_box = convert_rois_to_box()
        self.encoder = TransformerEncoder()
        self.backbone = Resnet_FPN_merge()

        # self.merge_head = merge_head()
        self.row_head = predict_head_row()
        self.col_head = predict_head_col()
        self.middle = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.flatten_fm = nn.Linear(in_features= 6272, out_features= 512)
    def forward(self, x , rois):
        if (torch.isnan(x).any()):
            print("error fw model ")
        rois_1 = torch.tensor(rois[0])
        rois_new = self.convert_rois_to_box(rois_1)
        x_embed, y_embed = self.pos_embedding(rois[0])
        if (torch.any(torch.isnan(x_embed))):
            print('pos error')
        elif  (torch.any(torch.isnan(y_embed))):
            print('pos error')
        pos_  = torch.cat((x_embed, y_embed), dim=-1) # (1, MxN, 512)
        p1, p2, p3, p4 = self.backbone(x) # p1: (1, 256, 128, 128)
        if  (torch.any(torch.isnan(p1))):
            print('feature error')
        feature_map = self.middle(p1) #  (1, 256, 128, 128)
        # print(f'feature_map :{feature_map.shape}')
        crops = roi_align(feature_map, rois_new) # (MxN, 128, 7, 7)  
        if  (torch.any(torch.isnan(crops))):
            print(' roi align error ')
        embedded_patches = crops.reshape(-1, 128*7*7) # (MxN, 6272)
        encoded_patches = self.flatten_fm(embedded_patches)# (MxN, 512)
        encoded_patches = torch.unsqueeze(encoded_patches, dim=0) # (1, MxN, 512)
        encode = pos_ + encoded_patches # (1, MxN , 512)
        # encode = encoded_patches

        x = self.encoder(encode) # (1, MxN, 512)
        if  (torch.any(torch.isnan(x))):
            print(' encoder error ')        
        row_logits = self.row_head(x)
        col_logits = self.col_head(x)
        col_logits = col_logits.view(-1)
        row_logits = row_logits.view(-1)
        return row_logits , col_logits

def get_pred_bbox_list(row_list_final, col_list_final):
    row_bbox_list = []
    for index in range(len(row_list_final) - 1):
        row_bbox_list.append([row_list_final[index], row_list_final[index + 1]])
    col_bbox_list = []
    for index in range(len(col_list_final) - 1):
        col_bbox_list.append([col_list_final[index], col_list_final[index + 1]])
    # print(row_bbox_list)
    # print(col_bbox_list)
    cell_bbox_list = []
    for row_idx , row_bbox in enumerate(row_bbox_list):
        for col_idx, col_bbox in enumerate(col_bbox_list):
            x1 = col_bbox[0]
            y1 = row_bbox[0]
            x2 = col_bbox[1]
            y2 = row_bbox[1]
            updated_bbox = [row_idx, col_idx, x1, y1, x2, y2]
            cell_bbox_list.append(updated_bbox)    
    return cell_bbox_list

def get_final_pred_merge(row_pred, col_pred, num_row, num_col):
    pred_col = col_pred.cpu()
    mask_1 = pred_col > 0.9
    mask_0 = pred_col <=0.9
    final_pred_col = np.zeros(shape = np.shape(pred_col) , dtype= int)
    final_pred_col[np.where(mask_1 == True)] = 1 
    final_pred_col = np.reshape(final_pred_col, (num_row + 1, num_col + 1))

    pred_row = row_pred.cpu()
    mask_1 = pred_row > 0.9
    mask_0 = pred_row <=0.9
    final_pred_row = np.zeros(shape = np.shape(pred_row) , dtype= int)
    final_pred_row[np.where(mask_1 == True)] = 1
    final_pred_row = np.reshape(final_pred_row, (num_row + 1, num_col + 1))

    return final_pred_row, final_pred_col

def get_col_spaning_cell(gt_col):
    # gt_col = gt['col'][0]
    list_merge_col = {}
    for row_idx in range(gt_col.shape[0]):  
        list_merge_col[row_idx] = []
    list_merge_col
    flag = 0

    for row_idx in range(gt_col.shape[0]):
        col_tmp = gt_col[row_idx]
        for col_idx in range(1,len(col_tmp)):
            if (col_tmp[col_idx] == 1 and col_tmp[col_idx-1]==0): # start counting 
                flag = 1
                start_col = col_idx - 1
            elif (col_tmp[col_idx] == 0 and col_tmp[col_idx-1]==1): # end counting
                flag = 0
                end_col = col_idx - 1
                # list_merge_col[row_idx].append([start_col,end_col])
                list_merge_col[row_idx].append([i for i in range(start_col,end_col+1)])
            
            if (flag == 1 and col_idx == len(col_tmp)-1): # end counting
                flag = 0
                end_col = col_idx
                # list_merge_col[row_idx].append([start_col,end_col])
                list_merge_col[row_idx].append([i for i in range(start_col,end_col+1)])
    return list_merge_col

def get_row_spaning_cell(gt_row):
    list_merge_row = {}
    for row_idx in range(gt_row.shape[1]):  
        list_merge_row[row_idx] = []
    list_merge_row
    flag  = 0
    for col_idx in range(gt_row.shape[1]):
        row_tmp = gt_row[:,col_idx]
        for row_idx in range(1,len(row_tmp)):
            if (row_tmp[row_idx] == 1 and row_tmp[row_idx-1]==0): # start counting
                flag = 1
                start_row = row_idx - 1
            elif (row_tmp[row_idx] == 0 and row_tmp[row_idx-1]==1): # end counting
                flag = 0
                end_row = row_idx - 1
                # list_merge_row[col_idx].append([start_row,end_row])
                list_merge_row[col_idx].append([i for i in range(start_row,end_row+1)])
            
            if (flag == 1 and row_idx == len(row_tmp)-1): # end counting
                flag = 0
                end_row = row_idx
                # list_merge_row[col_idx].append([start_row,end_row])
                list_merge_row[col_idx].append([i for i in range(start_row,end_row+1)])
    return list_merge_row
def check_single_cell_col(idx_row, idx_col, list_merge_col):
    span_cell = list_merge_col[idx_row]
    if len(span_cell) == 0:
        return True
    elif len(span_cell) > 0:
        if any(idx_col in sublist for sublist in span_cell):
            return False
        else :
            return True

def check_single_cell_row(idx_row, idx_col, list_merge_row):
    span_cell = list_merge_row[idx_col]
    if len(span_cell) == 0:
        return True
    elif len(span_cell) > 0:
        if any(idx_row in sublist for sublist in span_cell):
            return False
        else :
            return True

def get_bbox_location(idx_row_in, idx_col_in, bbox):
    for i in range(len(bbox)):
        row_idx_gt, col_idx_gt, x1, y1, x2, y2 = bbox[i]
        if (row_idx_gt == idx_row_in and col_idx_gt == idx_col_in):
            return [x1, y1, x2, y2]
    return -1

def get_bbox_spaning_col(idx_row, idx_col_list, bbox):
    bbox_list = []
    for idx_col in idx_col_list:
        try:
            bbox_location = get_bbox_location(idx_row, idx_col, bbox)
            bbox_list.append(bbox_location)
        except:
            pass
    bbox_array = np.array(bbox_list)
    # print(bbox_array.shape)
    x1 = np.min(bbox_array[:,0])
    y1 = np.min(bbox_array[:,1])
    x2 = np.max(bbox_array[:,2])
    y2 = np.max(bbox_array[:,3])

    final_bbox_span = [x1, y1, x2, y2]
    return final_bbox_span

def get_bbox_spaning_row(idx_row_list, idx_col, bbox):
    bbox_list = []
    for idx_row in idx_row_list:
        try:
            bbox_location = get_bbox_location(idx_row, idx_col, bbox)
            bbox_list.append(bbox_location)
        except:
            pass
    bbox_array = np.array(bbox_list)
    x1 = np.min(bbox_array[:,0])
    y1 = np.min(bbox_array[:,1])
    x2 = np.max(bbox_array[:,2])
    y2 = np.max(bbox_array[:,3])

    final_bbox_span = [x1, y1, x2, y2]
    return final_bbox_span

def get_table(gt_row, gt_col, bbox):
    merge_col = get_col_spaning_cell(gt_col)
    merge_row = get_row_spaning_cell(gt_row)
    table_dict = []

    # get span col cell
    for idx_row in range(len(merge_col)):
        merge_col_tmp = merge_col[idx_row]
        if len(merge_col_tmp) > 0:
            cell_merge = []
            
            for i in range(len(merge_col_tmp)):
                cell_dict = dict()
                list_col_merge_tmp = merge_col_tmp[i]
                bbox_spaning_col = get_bbox_spaning_col(idx_row, list_col_merge_tmp, bbox)
                cell_dict['row'] = idx_row
                cell_dict['col'] = list_col_merge_tmp
                cell_dict['bbox'] = bbox_spaning_col
                cell_dict['content'] = ''
                table_dict.append(cell_dict)   
    # get span row cell
    for idx_col in range(len(merge_row)):
        merge_row_tmp = merge_row[idx_col]
        if len(merge_row_tmp) > 0:
            cell_merge = []
            for i in range(len(merge_row_tmp)):
                cell_dict = dict()
                list_row_merge_tmp = merge_row_tmp[i]
                bbox_spaning_row = get_bbox_spaning_row(list_row_merge_tmp, idx_col, bbox)
                cell_dict['row'] = list_row_merge_tmp
                cell_dict['col'] = idx_col
                cell_dict['bbox'] = bbox_spaning_row
                cell_dict['content'] = ''
                table_dict.append(cell_dict)    

    # get single cell 
    for idx in range(len(bbox)):
        idx_row, idx_col, x1, y1, x2, y2 = bbox[idx]
        try:
            single_cell_row  = check_single_cell_row(idx_row, idx_col, merge_row)
            single_cell_col  = check_single_cell_col(idx_row, idx_col, merge_col)
            if single_cell_row == True  and single_cell_col == True:
                cell_dict = dict()
                cell_dict['row'] = [idx_row]
                cell_dict['col'] = [idx_col]
                cell_dict['bbox'] = [x1,y1,x2,y2]
                cell_dict['content'] = ''
                table_dict.append(cell_dict)
        except:
            print(idx_row)

    return table_dict

def show_final_result_v2(image_input , table_dict , h_original, w_original):
    # return to original image
    image_original = image_input.copy()
    for cell in table_dict:
        bbox = cell['bbox']
        x1_512 , y1_768 , x2_512, y2_768 = bbox
        x1_ori = int(x1_512/W_SHAPE * w_original)
        y1_ori = int(y1_768/H_SHAPE * h_original)
        x2_ori = int(x2_512/W_SHAPE * w_original)
        y2_ori = int(y2_768/H_SHAPE * h_original)
        cv2.rectangle(image_original, (x1_ori, y1_ori), (x2_ori, y2_ori), (255, 0, 0), 2)
    # plt.imshow(image_original)
    return image_original
def init_merge_model():

    model = Model_merge()
    return model    

H_SHAPE = 768
W_SHAPE = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    model = init_merge_model()
    print("model merge duoc khoi tao thanh cong")