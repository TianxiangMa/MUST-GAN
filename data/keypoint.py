import os.path
from data.base_dataset import BaseDataset, get_transform
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import torch


class KeyDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == 'train':
            self.dir_P = os.path.join(opt.dataroot, opt.phase + '_highres256')
            self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')
            self.dir_conn_map = os.path.join(opt.dataroot, 'pose_connect_map')
        elif opt.phase == 'test':
            self.dir_P = os.path.join(opt.dataroot, opt.phase + '_highres256')
            self.dir_K = os.path.join(opt.dataroot, opt.phase + 'K')
            self.dir_conn_map = os.path.join(opt.dataroot, 'pose_connect_map')

        self.dir_SP = opt.dirSem
        self.SP_input_nc = opt.SP_input_nc

        if opt.phase == 'train':
            self.init_categories_train(opt.unpairLst)
        elif opt.phase == 'test':
            self.init_categories_test(opt.pairLst)
        
        self.transform = get_transform(opt)

    def init_categories_train(self, unpairLst):
        pairs_file_train = pd.read_csv(unpairLst)
        self.size = len(pairs_file_train)
        self.imgs = []
        print('Loading data unpairs ...')
        for i in range(self.size):
            img = pairs_file_train.iloc[i]['images_name']
            self.imgs.append(img)

        print('Loading data unpairs finished ...')

    def init_categories_test(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        self.size = len(pairs_file_train)
        self.imgs = []
        print('Loading data pairs ...')
        for i in range(self.size):
            img = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            self.imgs.append(img)

        print('Loading data pairs finished ...')


    def __getitem__(self, index):
        if self.opt.phase == 'train':
            # person image
            P1_name = self.imgs[index]
            P1_path = os.path.join(self.dir_P, P1_name)
            P1_img = Image.open(P1_path).convert('RGB')
            P1_img = P1_img.resize((176, 256))
            P1 = self.transform(P1_img)

            # pose
            BP2_path = os.path.join(self.dir_K, P1_name + '.npy')
            BP2_img = np.load(BP2_path)
            PCM2_path = os.path.join(self.dir_conn_map, P1_name + '.npy')
            PCM2_mask = np.load(PCM2_path)
            BP2 = torch.from_numpy(BP2_img).float() #h,w,c
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 
            PCM2_mask = torch.from_numpy(PCM2_mask).float()
            BP2 = torch.cat([BP2, PCM2_mask], 0)

            # semantic
            SP1_name = self.split_name(P1_name, 'semantic_merge3')
            SP1_path = os.path.join(self.dir_SP, SP1_name)
            SP1_path = SP1_path[:-4] + '.npy'
            SP1_data = np.load(SP1_path)
            SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
            parti = np.random.randint(1, 8)
            for id in range(self.SP_input_nc):
                if id == 6 or id == 7: # arms and legs
                    if np.random.random() > 0.7:
                        continue
                SP1[id] = (SP1_data == id).astype('float32')

            return {'P1': P1, 'SP1': SP1, 'BP2': BP2, 'P1_path': P1_name}

        elif self.opt.phase == 'test':
            # person image
            P1_name, P2_name = self.imgs[index]
            P1_path = os.path.join(self.dir_P, P1_name)
            P1_img = Image.open(P1_path).convert('RGB')
            P2_path = os.path.join(self.dir_P, P2_name)
            P2_img = Image.open(P2_path).convert('RGB')
            P1_img = P1_img.resize((176, 256))
            P2_img = P2_img.resize((176, 256))
            P1 = self.transform(P1_img)
            P2 = self.transform(P2_img)

            # pose
            BP2_path = os.path.join(self.dir_K, P2_name + '.npy')
            PCM2_path = os.path.join(self.dir_conn_map, P2_name + '.npy')
            PCM2_mask = np.load(PCM2_path)
            BP2_img = np.load(BP2_path)
            BP2 = torch.from_numpy(BP2_img).float()
            BP2 = BP2.transpose(2, 0) #c,w,h
            BP2 = BP2.transpose(2, 1) #c,h,w 
            PCM2_mask = torch.from_numpy(PCM2_mask).float()
            BP2 = torch.cat([BP2, PCM2_mask], 0)

            # semantic
            SP1_name = self.split_name(P1_name, 'semantic_merge3')
            SP1_path = os.path.join(self.dir_SP, SP1_name)
            SP1_path = SP1_path[:-4] + '.npy'
            SP1_data = np.load(SP1_path)
            SP1 = np.zeros((self.SP_input_nc, 256, 176), dtype='float32')
            for id in range(self.SP_input_nc):
                SP1[id] = (SP1_data == id).astype('float32')

            return {'P1': P1, 'SP1': SP1, 'P2': P2, 'BP2': BP2, 'P1_path': P1_name, 'P2_path': P2_name}


    def __len__(self):
        if self.opt.phase == 'train':
            return self.size
        elif self.opt.phase == 'test':
            return self.size

    def name(self):
        return 'KeyDataset'

    def split_name(self,str,type):
        list = []
        list.append(type)
        if (str[len('fashion'):len('fashion') + 2] == 'WO'):
            lenSex = 5
        else:
            lenSex = 3
        list.append(str[len('fashion'):len('fashion') + lenSex])
        idx = str.rfind('id0')
        list.append(str[len('fashion') + len(list[1]):idx])
        id = str[idx:idx + 10]
        list.append(id[:2]+'_'+id[2:])
        pose = str[idx + 10:]
        list.append(pose[:4]+'_'+pose[4:])

        head = ''
        for path in list:
            head = os.path.join(head, path)
        return head

