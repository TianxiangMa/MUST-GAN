import os.path
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from getPoseMask import getPoseMask
import PIL
import random
import pandas as pd
import numpy as np
import torch
import json


root_path = '/hd1/matianxiang/MUST/datasets/' # Using yourself root
save_dir = os.path.join(root_path, 'pose_connection_map/')
os.makedirs(save_dir, exist_ok=True)

annotation_file_train = pd.read_csv(os.path.join(root_path, 'fashion-resize-annotation-train.csv'), sep=':')
annotation_file_test = pd.read_csv(os.path.join(root_path, 'fashion-resize-annotation-test.csv'), sep=':')
annotation_file = pd.concat([annotation_file_train, annotation_file_test], axis=0)

for i in range(len(annotation_file)):
    print('processing %d / %d ...' %(i, len(annotation_file)))
    B_row = annotation_file.iloc[i]
    y_cords = json.loads(B_row['keypoints_y'])
    x_cords = json.loads(B_row['keypoints_x'])
    B_kp_array = np.array([y_cords, x_cords])
    BP1_mask = getPoseMask(B_kp_array, 256, 176, radius=1, mode='Solid')
    name = B_row['name']
    print(name)
    np.save(os.path.join(save_dir, name + '.npy'), BP1_mask)
