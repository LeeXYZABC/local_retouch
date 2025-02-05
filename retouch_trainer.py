
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data as data_utils

import utils
import loss_utils
from unet import UNet
from retouch_datasets import RetouchDataset

from msglogger import logger, init_log

base_dir = os.path.split(os.path.realpath(__file__))[0]
log_path = os.path.join(base_dir, "./log")
if not os.path.exists(log_path):
    os.makedirs(log_path)
init_log(log_path, "retouch")


degree=1

device = "cuda"
eps = 1e-8
weight_decay = 0.5e-4
initial_learning_rate = 5e-4


class Loss:
    def __init__(self, device="cpu"):
        self.device = device
        self.lpips = loss_utils.LPIPSLoss().to(device)
        self.criterion = nn.MSELoss().to(device)
        self.tv = loss_utils.WeightedTVLoss().to(device)

    def mg_process(self, pred_mg_list, roi_input, roi_gt):
        mse_loss, lpips_loss, tv_loss = None, None, None
        roi_input = (roi_input + 1.0) / 2.
        
        for pred_index, pred_mg in enumerate(pred_mg_list):
            # for each blend layer
            pred_mg = (pred_mg - 0.5) * degree + 0.5
            pred_mg = pred_mg.clamp(0.0, 1.0)
            input_curr = F.interpolate(roi_input, pred_mg.shape[2:], mode='bilinear')
            # https://www.pegtop.net/delphi/articles/blendmodes/softlight.htm
            # f(a,b) =	2 * a * b + a2 * (1 - 2 * b) (for b < ½) sqrt(a) * (2 * b - 1) + (2 * a) * (1 - b) (else)
            pred_curr = (1 - 2 * pred_mg) * input_curr * input_curr \
                    + 2 * pred_mg * input_curr  # value: 0~1
            gt_curr = F.interpolate(roi_gt, pred_mg.shape[2:], mode='bilinear')
            
            curr_mse_loss = self.criterion(pred_curr, gt_curr)
            mse_loss = mse_loss + curr_mse_loss if mse_loss else curr_mse_loss
            
            if pred_index == len(pred_mg_list) - 1:
                lpips_loss = self.lpips(pred_curr.float(), gt_curr.float())
            
            curr_tv_loss = self.tv(pred_curr.float(), gt_curr.float())
            tv_loss = tv_loss + curr_tv_loss if tv_loss else curr_tv_loss
            
        return mse_loss, lpips_loss, tv_loss


def run_epoch(
    data_loader, model, optimizer, checkpoint_dir,
    global_step, curr_epoch, 
    loss_module
):
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.train()
        optimizer.zero_grad()

        roi_input = batch["roi_input"]
        roi_gt = batch["roi_gt"]

        roi_input = roi_input.to(device)
        roi_gt = roi_gt.to(device)

        roi_input = utils.preprocess_roi(roi_input)
        pred_mg, pred_mg1, pred_mg2, pred_mg3, pred_mg4 = model(roi_input)   
        mse_loss, lpips_loss, tv_loss = loss_module.mg_process(
                [pred_mg, pred_mg1, pred_mg2, pred_mg3, pred_mg4], 
                roi_input, roi_gt)

        mse_weight = 0.8
        lpips_weight = 0.9
        tv_weight = 1
        loss = mse_weight * mse_loss + lpips_weight * lpips_loss + tv_weight * tv_loss
            
        loss.backward()
        optimizer.step()

        if i % 100 == 0:       
            logger.info("global_step: {}, curr_epoch: {}, mse_loss: {}, tv_loss: {}, lpips_loss: {}".format(
                        global_step, curr_epoch, 
                        round(mse_loss.item(), 5),
                        round(tv_loss.item(), 5),
                        round(lpips_loss.item(), 5)
                    )
                 )
            
        if global_step % 5000 == 0:
            utils.save_checkpoint(model, optimizer, global_step, 
                                  checkpoint_dir, curr_epoch, 
                                  prefix="unet_")

        global_step += 1
        
    return global_step

def run_training(
    train_data_loader, model, optimizer, checkpoint_dir, 
    global_step, start_epoch, total_epoches):
    
    loss_module = Loss(device)
    
    for curr_epoch in range(start_epoch, total_epoches):        
        if curr_epoch > 20:
            utils.adjust_lr(
                optimizer, initial_learning_rate, 
                curr_epoch, total_epoches)
        
        global_step = run_epoch(
            train_data_loader, model, optimizer, checkpoint_dir, 
            global_step, curr_epoch, 
            loss_module)


def run_main():
    
    global_step = 0
    global_epoch = 0
    batch_size = 8
    total_epoches = 60

    checkpoint_dir = "./checkpoints"
    checkpoint_path = None # "checkpoints/unet_checkpoint_step000375000.pth"
    
    ffhq_folder = "/nas01/gpu04_backup/gpu07/tmp/data/ffhq/ffhq_1024_zip/tmp"
    ffhqr_folder = "/nas01/gpu04_backup/gpu07/tmp/data/ffhq/ffhqr/ffhqr"

    model = UNet(3, 3, deep_supervision=True).to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=initial_learning_rate, 
        betas=(0.5, 0.999), 
        weight_decay=weight_decay, 
        eps=eps
    )
    
    if checkpoint_path:
        model, optimizer, global_step, global_epoch = utils.load_checkpoint(
                checkpoint_path, model, optimizer, 
                overwrite_global_states=True,
                use_cuda=("cuda" in device))
    train_dataset = RetouchDataset(ffhq_folder, ffhqr_folder)    
    start_epoch = global_step // (len(train_dataset)//batch_size)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    run_training(train_data_loader, model, optimizer, checkpoint_dir,
        global_step, start_epoch, total_epoches)



if __name__ == "__main__":
    run_main()