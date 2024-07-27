import os, random, cv2, argparse
import torch
from torch.utils import data as data_utils
from os.path import dirname, join, basename, isfile
import numpy as np
from glob import glob
import torchvision.utils as vutils
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm
import ast
import json
import re
import heapq
from PIL import Image
syncnet_T = 1
RESIZED_IMG = 256

# connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),(7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),(13,14),(14,15),(15,16),  # 下颌线
#                        (17, 18), (18, 19), (19, 20), (20, 21), #左眉毛
#                        (22, 23), (23, 24), (24, 25), (25, 26), #右眉毛
#                        (27, 28),(28,29),(29,30),# 鼻梁
#                        (31,32),(32,33),(33,34),(34,35), #鼻子
#                        (36,37),(37,38),(38, 39), (39, 40), (40, 41), (41, 36), # 左眼
#                        (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), # 右眼
#                        (48, 49),(49, 50), (50, 51),(51, 52),(52, 53), (53, 54), # 上嘴唇 外延
#                        (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # 下嘴唇 外延
#                        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),  (65, 66), (66, 67), (67, 60) #嘴唇内圈
#               ]  


def get_image_list(data_root, split):
    filelist = []
    imgNumList = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                filename = line.split('\t')[0]
                imgNum = int(line.split('\t')[1])
                filelist.append(os.path.join(data_root, filename))
                imgNumList.append(imgNum)
    return filelist, imgNumList



class Dataset(object):
    def __init__(self, 
                 data_root, 
                 split, 
                 use_audio_length_left=1,
                 use_audio_length_right=1,
                 whisper_model_type = "tiny"
                 ):
        self.all_videos, self.all_imgNum = get_image_list(os.path.join(data_root,'images'), split)
        self.audio_feature = [use_audio_length_left,use_audio_length_right]
        self.all_img_names = []
        self.split = split
        self.whisper_path = os.path.join(data_root, 'audios')
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )   


        for vidname in tqdm(self.all_videos, desc="Preparing dataset"):
            img_names = glob(os.path.join(vidname, '*.png'))
            img_names.sort(key=lambda x:int(x.split("/")[-1].split('.')[0]))
            self.all_img_names.append(img_names)
            
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])


    def __len__(self):
        return len(self.all_videos)


    def __getitem__(self, idx):
        idx = random.randint(0, len(self.all_videos) - 1)
        #随机选择某个video里
        part2 = self.all_videos[idx].split('/')[2]
        part3 = self.all_videos[idx].split('/')[3]
        part4 = self.all_videos[idx].split('/')[4]
        sub_folder_name = os.path.join(part2, part3, part4)
        video_imgs = self.all_img_names[idx]
        img_idx = random.randint(0,len(video_imgs)-1)
        img_name = video_imgs[img_idx]
        random_element = random.randint(0,len(video_imgs)-1)
        while abs(random_element - img_idx) <= 5:
            random_element = random.randint(0,len(video_imgs)-1)
        ref_image_name = video_imgs[random_element]
        image = self.transform(Image.open(img_name).convert('RGB').resize((RESIZED_IMG, RESIZED_IMG)))
        ref_frame = self.transform(Image.open(ref_image_name).convert('RGB').resize((RESIZED_IMG, RESIZED_IMG)))
        mask = torch.zeros(ref_frame.shape[1], ref_frame.shape[2])
        mask[:ref_frame.shape[2]//2,:] = 1
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        masked_image = image * (mask > 0.5)
      
        #音频特征
        audio_index = self.get_frame_id(img_name)
        audio_feat_path = os.path.join(self.whisper_path, sub_folder_name, str(audio_index).zfill(6) + ".npy")
        audio_feature = np.load(audio_feat_path)  
        audio_feature = torch.squeeze(torch.FloatTensor(audio_feature))
        sample = {}
        sample['ref_image'] = ref_frame
        sample['image'] = image
        sample['masked_image'] = masked_image
        sample['mask'] = mask
        sample['audio_feature'] = audio_feature
         
        return sample
         
def collate_fn(data):
    ref_image = torch.stack([example['ref_image'] for example in data])
    image = torch.stack([example['image'] for example in data])
    masked_image = torch.stack([example['masked_image'] for example in data])
    mask = torch.stack([example['mask'] for example in data])
    audio_feature = torch.stack([example['audio_feature'] for example in data])
    return ref_image, image, masked_image, mask, audio_feature

    
if __name__ == "__main__":
    data_root = 'data'
    val_data = Dataset(data_root, 
                          'val', 
                          use_audio_length_left = 2,
                          use_audio_length_right = 2,
                          whisper_model_type = "tiny"
                          )
    val_data_loader = data_utils.DataLoader(
        val_data, batch_size=3, shuffle=True,collate_fn=collate_fn,
        num_workers=1)
    print("val_dataset:",val_data_loader.__len__())

    for i, data in enumerate(val_data_loader):
        ref_image, image, masked_image, mask, audio_feature = data
        print("ref_image: ", ref_image.shape, image.shape,  masked_image.shape, mask.shape, audio_feature.shape)

 
