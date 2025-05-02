import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
import cv2
import ast


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

class Resnet_FPN_split(nn.Module):
    def __init__(self,out_channels = 128):
        super(Resnet_FPN_split, self).__init__()
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

class transferlayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transferlayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv(x)
        return x

class mess_passing_H(nn.Module):
    def __init__(self, in_channels = 32, out_channels= 32):
        super(mess_passing_H, self).__init__()
        self.t2d_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[1,9], stride=(1,1), padding=(0,4))
        self.d2t_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[1,9], stride=(1,1), padding=(0,4))
        self.relu = nn.ReLU()
    def forward(self, x):
        nB, c, h, w = x.shape
        feature_list_old = []   
        feature_list_new = []
        for cnt in range(h):
            feature_list_old.append(x[:,:,cnt,:].unsqueeze(dim = 2))
        feature_list_new.append(x[:,:,0,:].unsqueeze(dim = 2))
        feature = self.t2d_layers(feature_list_old[0])
        feature = self.relu(feature) + feature_list_old[1]
        feature_list_new.append(feature)
        for cnt in range(2,h):
            feature =  self.t2d_layers(feature_list_new[cnt-1])
            feature = self.relu(feature) + feature_list_old[cnt]
            feature_list_new.append(feature)
        feature_list_new.append(feature)
        feature_list_old = feature_list_new
        feature_list_new = []
        length = h - 1 
        feature_list_new.append(feature_list_old[length])
        feature = self.d2t_layers(feature_list_old[length])
        feature = self.relu(feature) + feature_list_old[length - 1]
        feature_list_new.append(feature)
        for cnt in range(2, h):
            feature = self.d2t_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[length - cnt]
            feature_list_new.append(feature)
        feature_list_new.reverse()
        processed_feature = torch.stack(feature_list_new,dim = 2)
        processed_feature = processed_feature.squeeze(dim = 3 )
        return processed_feature

class mess_passing_W(nn.Module): # L -> R
    def __init__(self, in_channels = 32, out_channels= 32):
        super(mess_passing_W, self).__init__()
        self.l2r_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[9,1], stride=(1,1), padding=(4,0))
        self.r2l_layers = nn.Conv2d(in_channels, out_channels, kernel_size=[9,1], stride=(1,1), padding=(4,0))
        self.relu = nn.ReLU()
    def forward(self, x):
        nB, c, h, w = x.shape
        feature_list_old = []
        feature_list_new = []
        for cnt in range(w):
            feature_list_old.append(x[:, :, : , cnt].unsqueeze(dim = 3))
        feature_list_new.append(x[:, :, :, 0].unsqueeze(dim = 3))
        feature = self.l2r_layers(feature_list_old[0])
        feature = self.relu(feature) + feature_list_old[1]
        feature_list_new.append(feature)
        for cnt in range(2, w):
            feature = self.l2r_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[cnt]
            feature_list_new.append(feature)
        feature_list_old = feature_list_new
        feature_list_new = []
        length = w - 1
        feature_list_new.append(feature_list_old[length])
        feature = self.r2l_layers(feature_list_old[length])
        feature = self.relu(feature) + feature_list_old[length - 1]
        feature_list_new.append(feature)
        for cnt in range(2, w):
            feature = self.r2l_layers(feature_list_new[cnt - 1])
            feature = self.relu(feature) + feature_list_old[length - cnt]
            feature_list_new.append(feature)
        feature_list_new.reverse()
        processed_feature = torch.stack(feature_list_new, dim=3)
        processed_feature = torch.squeeze(processed_feature, axis=4)
        return processed_feature
        
class sun_branch_1(nn.Module):
    def __init__(self, size_):
        super(sun_branch_1, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

class sun_branch_2(nn.Module):
    def __init__(self, size_):
        super(sun_branch_2, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

class sun_branch_3(nn.Module):
    def __init__(self, size_):
        super(sun_branch_3, self).__init__()
        self.size_ = size_
        self.Max = nn.MaxPool2d(kernel_size=size_, stride=size_ , padding= (0,0))
        self.Conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 1)
        self.relu = nn.ReLU()
    def forward(self, x , input_channel):
        x = x.clone()
        x = self.Max(x)
        x = self.Conv(x)
        x = self.relu(x)
        return x

class updasample_combines(nn.Module):
    def __init__(self ):
        super(updasample_combines, self).__init__()
        self.conv = nn.Conv2d(32, 256, kernel_size=3, stride=1, padding= 'same')
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x_tensor):
        x = F.interpolate(x_tensor, size =(H_SHAPE,W_SHAPE), mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class predict_row(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(predict_row, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        row_ = self.conv(x)
        final_row = self.sigmoid(row_)
        return final_row

class predict_col(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(predict_col, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        col_ = self.conv(x)
        final_col = self.sigmoid(col_)
        return final_col

class model_split(nn.Module):
    def __init__(self):
        super(model_split, self).__init__()  # 128 -> 256
        self.backbone = Resnet_FPN_split()
        self.mess_passing_H = mess_passing_H()
        self.mess_passing_W = mess_passing_W()
        self.predict_row = predict_row(256, 1)
        self.predict_col = predict_col(256, 1)
        self.sun_branch_W_1 = sun_branch_1(size_=(1,2))
        self.sun_branch_W_2 = sun_branch_2(size_=(1,2))
        self.sun_branch_W_3 = sun_branch_3(size_=(1,2))

        self.sun_branch_H_1 = sun_branch_1(size_=(2,1))
        self.sun_branch_H_2 = sun_branch_2(size_=(2,1))
        self.sun_branch_H_3 = sun_branch_3(size_=(2,1))

        self.transfer_layer = transferlayer(128, 64)  # 256 -> 16
        self.updasample_combines_H = updasample_combines()
        self.updasample_combines_W = updasample_combines()  # 256 * 256
    def forward(self, image):
        c2, c3 , c4, c5 = self.backbone(image)
        fm = self.transfer_layer(c2)
        x_H = fm
        x_W = fm
        # for _ in range(3):
        x_H= self.sun_branch_H_1(x_H, x_H.shape[1])
        x_H= self.sun_branch_H_2(x_H, x_H.shape[1])
        x_H= self.sun_branch_H_3(x_H, x_H.shape[1])

        x_W = self.sun_branch_W_1(x_W, x_W.shape[1])
        x_W = self.sun_branch_W_2(x_W, x_W.shape[1])
        x_W = self.sun_branch_W_3(x_W, x_W.shape[1])
        # print(f'x_W : {x_W.shape}')
        col_feature = self.mess_passing_H(x_H)
        col_feature = self.updasample_combines_H(col_feature)  # 256 * 256

        row_feature = self.mess_passing_W(x_W)
        row_feature = self.updasample_combines_W(row_feature)  # 256 * 256
        # print(f'row_feature :{row_feature.shape}, col_feature:{col_feature.shape}')
        final_row = self.predict_row(row_feature)
        final_col = self.predict_col(col_feature)
        return final_row, final_col
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
H_SHAPE = 768
W_SHAPE = 512
def init_model():
    model = model_split()
    return model
def contour_dis(contour_1, contour_2):
    min_dis = 99999
    for point1 in contour_1:
        for point2 in contour_2:
            dis = np.linalg.norm(point1[0] - point2[0])
            if dis < min_dis:
                min_dis = dis
    return min_dis

def merge_countour(contour_1, contour_2):
    merge_c = np.vstack((contour_1, contour_2))
    merge_c = cv2.convexHull(merge_c)
    return merge_c

def is_near_border(contour , mode):
    if mode == 'row':
        for point in contour:
            x, y = point[0]
            if y <= 2 :
                return True
            if y >= H_SHAPE-2:
                return True
        return False
    elif mode == 'col':
        for point in contour:
            x, y = point[0]
            if x <= 2:
                return True
            if x >= W_SHAPE-2:
                return True
        return False

def get_col_pred_line(img_col_pred , threshold_heatmap_col , threshold_contour_col):
  test_image = img_col_pred[0]
  test_image =   test_image.permute(1,2,0)
  test_image = test_image.cpu()
  test_image_np = test_image.detach().numpy()
  col_img_pred = test_image_np.copy()
  col_img_pred = np.squeeze(col_img_pred, axis= 2)
  reversed_binary_img = 1 - col_img_pred
  col_mean = np.mean(col_img_pred , axis = 0)
  mask_1 = col_mean> threshold_heatmap_col
  mask_0 = col_mean <= threshold_heatmap_col
  col_mean[mask_1] = 0
  col_mean[mask_0] = 1
  image_finale_col_pred = np.tile(col_mean, (512, 1))
  contours, hierarchy = cv2.findContours(image_finale_col_pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
  # remove contour near border
  contour_after_remove = []
  for contour_mem in contours:
    if not is_near_border(contour_mem, 'col'):
      contour_after_remove.append(contour_mem)

  final_countours = []    
  used = set()
  for i, contour_1 in enumerate(contour_after_remove):
    if i in used:
      continue
    for j, contour_2 in enumerate(contour_after_remove):
      if i!= j and j not in used:
        dis = contour_dis(contour_1, contour_2)
        if dis < threshold_contour_col:
            contour_1 = merge_countour(contour_1, contour_2)
            used.add(j)
    final_countours.append(contour_1)
    used.add(i) 

  polynomial_fits = []
  for i, contour in enumerate(final_countours):
    x = contour[:, 0, 1]
    y = contour[:, 0, 0]
    try:
      polynomial_fit = np.polyfit(x, y, 0)
      polynomial_fits.append(polynomial_fit)
    except:
        pass
  final_list = []
  for line in polynomial_fits:
    line_int = int(line[0])
    final_list.append(line_int)
  return final_list

def get_row_pred_line(img_row_pred , threshold_heatmap_row , threshold_contour_row):
  test_image = img_row_pred[0]
  # print(test_image.shape)
  test_image =   test_image.permute(1,2,0)
  test_image = test_image.cpu()
  # test_image_np = np.array(test_image)
  test_image_np = test_image.detach().numpy()
  row_img_pred = test_image_np.copy()
  row_img_pred = np.squeeze(row_img_pred, axis= 2)
  reversed_binary_img = 1 - row_img_pred
  row_mean = np.mean(row_img_pred , axis = 1)
  mask_1 = row_mean> threshold_heatmap_row
  mask_0 = row_mean <= threshold_heatmap_row
  row_mean[mask_1] = 0
  row_mean[mask_0] = 1
  row_mean_reshape = row_mean.reshape(H_SHAPE,1)
  # print(row_mean)
  image_finale_row_pred = np.tile(row_mean_reshape, (1, W_SHAPE))
  # print(f' check {image_finale_row_pred.shape}')
  contours, hierarchy = cv2.findContours(image_finale_row_pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # remove contour near border 
  contour_after_remove = []
  for contour_mem in contours:
    if not is_near_border(contour_mem, 'row'):
      contour_after_remove.append(contour_mem)

  final_countours = []
  used = set()
  for i, contour_1 in enumerate(contour_after_remove):
    if i in used:
      continue
    for j, contour_2 in enumerate(contour_after_remove):
      if i!= j and j not in used:
        dis = contour_dis(contour_1, contour_2)
        if dis < threshold_contour_row:
            contour_1 = merge_countour(contour_1, contour_2)
            used.add(j)
    final_countours.append(contour_1)
    used.add(i) 

  polynomial_fits = []
  for i, contour in enumerate(final_countours):
    x = contour[:, 0, 1]
    y = contour[:, 0, 0]
    try:
      polynomial_fit = np.polyfit(y, x, 0)
      polynomial_fits.append(polynomial_fit)
    except:
      pass
  final_list = []
  for line in polynomial_fits:
    line_int = int(line[0])
    final_list.append(line_int)
  return final_list

def show_line_pred(row_list, col_list, image_np):
  image_input = cv2.resize(image_np, (W_SHAPE,H_SHAPE))
  for line in row_list:
    # print(line)
    y_coord = line
    start_point = (0, y_coord)
    end_point = (W_SHAPE, y_coord)
    cv2.line(image_input, start_point, end_point, (255,0,0),1)
    # print(line[0])
  for line in col_list:
    # print(line)
    x_coord = line
    start_point = (x_coord, 0)
    end_point = (x_coord, H_SHAPE)
    cv2.line(image_input, start_point, end_point, (255,0,0),1)
  return image_input
#   plt.imshow(image_input)
#   plt.show()

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

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set device to gpu or cpu
    model = init_model()
    print("model split da duoc khoi tao thanh cong")