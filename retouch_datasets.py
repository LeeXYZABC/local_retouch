import os
import cv2
import torch
import pickle
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F

import utils


def read_image(img_path):
    img = Image.open(img_path)
    img = np.array(img.convert('RGB'))
    img = img.astype(float)
    img = img / 255.
    return img

def rgb2gray(rgb):
    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]
    rgb[:, 0 ]= 0.299*R + 0.587*G + 0.114*B
    rgb = rgb[:, :1]
    return rgb

def preprocess(roi_input, roi_gt):    
    align_size = (512, 512)
    roi_input = F.interpolate(
        roi_input.unsqueeze(dim=0), 
        size=align_size, mode='bilinear', align_corners=True)
    roi_gt = F.interpolate(
        roi_gt.unsqueeze(dim=0), 
        size=align_size, mode='bilinear', align_corners=True)
    roi_input = roi_input[0]
    roi_gt = roi_gt[0]
    return {
        "roi_input": roi_input,
        "roi_gt": roi_gt,
    }


class RetouchDataset(object):
    def __init__(self, ffhq_folder=None, ffhqr_folder=None):
        super(RetouchDataset, self).__init__()
        self.ffhq_folder = ffhq_folder
        self.ffhq_r_folder = ffhqr_folder
        with open("./data/ffhq_bboxes.pkl", "rb") as f:
            self.ffhq_bboxes_maps = pickle.load(f)
        self.input_size = 512
        self.ffhq_r_paths = []
        self.ffhq_paths = []
        _ffhq_r_paths = utils.paths_from_folder(self.ffhq_r_folder)
        for ffhq_r_path in _ffhq_r_paths:
            filename = os.path.basename(ffhq_r_path)
            ffhq_path = os.path.join(self.ffhq_folder, str(filename))
            if ffhq_path in self.ffhq_bboxes_maps:
                self.ffhq_paths.append(str(ffhq_path))
                self.ffhq_r_paths.append(str(ffhq_r_path))
            
    def __getitem__(self, index):
        while 1:
            index = random.randint(0, len(self.ffhq_r_paths) - 1)
            ffhq_path = self.ffhq_paths[index]
            ffhq_r_path = self.ffhq_r_paths[index]
            ffhq_bboxes = self.ffhq_bboxes_maps[ffhq_path]
            
            if len(ffhq_bboxes) != 1:
                continue
                
            bbox = ffhq_bboxes[0]
            img_ffhq = read_image(ffhq_path)
            img_ffhq_r = read_image(ffhq_r_path)
            roi_origin = utils.get_roi(img_ffhq, bbox)
            roi_retouch = utils.get_roi(img_ffhq_r, bbox)
            roi_input, roi_gt = utils.img2tensor(
                [roi_origin, roi_retouch], bgr2rgb=True, float32=True)
            return preprocess(roi_input, roi_gt)
    
    def __len__(self):
        return len(self.ffhq_r_paths)


