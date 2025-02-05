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

def preprocess(roi_origin, roi_retouch, roi_skin):
    align_size = (256, 256)
    
    roi_skin = F.interpolate(
        roi_skin.unsqueeze(dim=0), 
        size=align_size, mode='bilinear', align_corners=True)
    image_origin= F.interpolate(
        roi_origin.unsqueeze(dim=0), 
        size=align_size, mode='bilinear', align_corners=True)
    image_retouch = F.interpolate(
        roi_retouch.unsqueeze(dim=0), 
        size=align_size, mode='bilinear', align_corners=True)

    skin_mask = rgb2gray(roi_skin)
    skin_mask = skin_mask[0]
    
    image_in = F.interpolate(
        roi_origin.unsqueeze(dim=0), 
        size=(768, 768), mode='bilinear', align_corners=True)
    image_in = utils.preprocess_roi(image_in)
    image_in = image_in[0]

    rgb_diff = torch.abs((image_retouch - image_origin) * 255.) * roi_skin
    diff = rgb2gray(rgb_diff)
    diff_low = (diff > 3.).float()
    diff_upper = (diff > 125.).float()
    diff_region = (255 - diff)  * (diff_low) * (1 - diff_upper)
    diff_region = diff_region / 255.
    diff_region = (diff_region > 0.5).float()
    mask_gt = diff_region[0]

    image_gt = F.interpolate(
        roi_retouch.unsqueeze(dim=0), 
        size=(512, 512), mode='bilinear', align_corners=True)
    image_gt = image_gt[0]
    
    return {
        'mask_gt': mask_gt, 
        'image_input': image_in, 
        "skin_mask": skin_mask,
        "image_gt": image_gt
    }


class JointDataset(object):
    def __init__(self, ffhq_folder=None, ffhqr_folder=None):
        super(JointDataset, self).__init__()
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
            if filename in self.ffhq_bboxes_maps:
                self.ffhq_paths.append(str(ffhq_path))
                self.ffhq_r_paths.append(str(ffhq_r_path))
            
    def __getitem__(self, index):
        while 1:
            index = random.randint(0, len(self.ffhq_r_paths) - 1)
            ffhq_path = self.ffhq_paths[index]
            ffhq_r_path = self.ffhq_r_paths[index]
            basename = os.path.basename(ffhq_path)
            ffhq_bboxes = self.ffhq_bboxes_maps[basename]
            
            if len(ffhq_bboxes) != 1:
                continue
                
            bbox = ffhq_bboxes[0]

            img_ffhq = read_image(ffhq_path)
            img_ffhq_r = read_image(ffhq_r_path)

            skin_path = os.path.join("data/skins", basename)
            skin_mask = read_image(skin_path)
                    
            roi_origin = utils.get_roi(img_ffhq, bbox)
            roi_retouch = utils.get_roi(img_ffhq_r, bbox)
            roi_skin = utils.get_roi(skin_mask, bbox)

            roi_origin, roi_retouch, roi_skin = utils.img2tensor(
                [roi_origin, roi_retouch, roi_skin], 
                bgr2rgb=True, float32=True)
            
            return preprocess(roi_origin, roi_retouch, roi_skin)
    
    def __len__(self):
        return len(self.ffhq_r_paths)


