import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data as data_utils

from msglogger import logger, init_log

base_dir = os.path.split(os.path.realpath(__file__))[0]
log_path = os.path.join(base_dir, "./log")
if not os.path.exists(log_path):
    os.makedirs(log_path)
init_log(log_path, "joint")

from modelscope.models.cv.skin_retouching.detection_model.detection_unet_in import \
    DetectionUNet
from modelscope.models.cv.skin_retouching.inpainting_model.inpainting_unet import \
    RetouchingNet

import utils
import loss_utils
from joint_datasets import JointDataset


device = "cuda"
eps = 1e-8
weight_decay = 0.5e-4
initial_learning_rate = 5e-4


class Loss:
    def __init__(self, device="cpu"):
        self.device = device
        self.lpips = loss_utils.LPIPSLoss().to(device)
        self.criterion = nn.MSELoss().to(device)
        self.l1 = nn.L1Loss(reduction='mean').to(device)
        self.tv = loss_utils.WeightedTVLoss().to(device)
        self.dice = loss_utils.DiceLoss().to(device)

    def calc_det_loss(self, pred, gt):
        dice_loss = self.dice(pred, gt)
        mse_loss = self.criterion(pred, gt)
        tv_loss = self.tv(pred, gt)
        loss = dice_loss + mse_loss + tv_loss 
        return loss, dice_loss, mse_loss, tv_loss 
        
    def calc_inp_loss(self, pred, gt):
        mse_loss = self.criterion(pred, gt)
        l1_loss = self.l1(pred, gt)
        tv_loss = self.tv(pred, gt)
        # lpips_loss = self.lpips(pred, gt)
        # loss = l1_loss + mse_loss + tv_loss + lpips_loss
        loss = l1_loss + tv_loss 
        return loss, l1_loss, tv_loss


def retouch_local(image, sub_mask_pred, inpainting_net):
    sub_H, sub_W = image.shape[2:]
    sub_mask_pred_hard_low = (sub_mask_pred >= 0.35).float()
    sub_mask_pred_hard_high = (sub_mask_pred >= 0.5).float()

    sub_mask_pred = sub_mask_pred * (
        1 - sub_mask_pred_hard_high) + sub_mask_pred_hard_high
    sub_mask_pred = sub_mask_pred * sub_mask_pred_hard_low
    sub_mask_pred = 1 - sub_mask_pred

    image_input = image * sub_mask_pred

    inpainting_output = inpainting_net(image_input, sub_mask_pred)
    inpainting_comp = image_input + (
                1 - sub_mask_pred) * inpainting_output
    return inpainting_comp


def run_epoch(
    data_loader, 
    model_det, optimizer_det, 
    model_inp, optimizer_inp, 
    checkpoint_dir,
    global_step, curr_epoch, 
    loss_module
):
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        model_det.train()
        optimizer_det.zero_grad()
        mask_gt = batch['mask_gt']
        image_input = batch['image_input']
        skin_mask = batch['skin_mask']
        mask_gt = mask_gt.to(device)
        image_input = image_input.to(device)
        skin_mask = skin_mask.to(device)

        align_size_list = [(256, 256)]
        align_size = random.sample(align_size_list, 1)[0]
    
        sub_H, sub_W = mask_gt.shape[2:]
        mask_pred = torch.sigmoid(model_det(image_input))
        mask_pred = F.interpolate(mask_pred, size=(sub_H, sub_W), mode='nearest')
        mask_pred = mask_pred
        mask_gt = mask_gt * skin_mask
        det_loss, dice_loss, mse_loss, tv_loss = loss_module.calc_det_loss(mask_pred, mask_gt)
        det_loss.backward()
        optimizer_det.step()
        
        if i % 1 == 0:
            model_inp.train()
            optimizer_inp.zero_grad()
            image_gt = batch['image_gt']
            image_gt = image_gt.to(device)
            image_gt = F.interpolate(image_gt, size=align_size, mode='bilinear', align_corners=True)
            # preprocess, rgb to [-1, 1]
            # image_gt = utils.preprocess_roi(image_gt)
            mask_input = F.interpolate(mask_pred, size=align_size, mode='bilinear', align_corners=True)
            mask_input = mask_input.clone().detach()
            image_gt = image_gt.clone().detach()
            image_pred = retouch_local(image_gt, mask_input, model_inp)
            # postproces [-1, 1] to rgb
            # image_pred = utils.postrocess_roi(image_pred)
            # image_gt = utils.postrocess_roi(image_gt)
            inp_loss, l1_loss, tv_loss = loss_module.calc_inp_loss(image_pred, image_gt)                
            inp_loss.backward()
            optimizer_inp.step()
            
        if i % 100 == 0:   
            logger.info("global_step: {}, curr_epoch: {}, det_dice_loss: {}, det_mse_loss: {}, det_tv_loss: {}, inp_l1_loss: {}, inp_tv_loss: {}".format(
                        global_step, curr_epoch, 
                        round(dice_loss.item(), 5),
                        round(mse_loss.item(), 5),
                        round(tv_loss.item(), 5),
                        round(l1_loss.item(), 5),
                        round(tv_loss.item(), 5)
                    )
                 )
            # image_gt = utils.postrocess_roi(image_gt)
            # mask_input = mask_input.detach()
            # image_pred = utils.postrocess_roi(image_pred.detach())
            import cv2
            def img_postprocess(img):
                img = img.detach()
                img = img.permute(0, 2, 3, 1)
                img = img * 255.
                img = img.cpu().numpy().astype(np.uint8)[0]
                h, w= img.shape[:2]
                return cv2.resize(img, (2*w, 2*h))
            cv2.imwrite("log/image_gt.png", img_postprocess(image_gt))
            cv2.imwrite("log/image_mask.png", img_postprocess(mask_input))
            cv2.imwrite("log/image_pred.png", img_postprocess(image_pred))

        
        if global_step % 5000 == 0:
            utils.save_checkpoint(model_det, optimizer_det, global_step, 
                                  checkpoint_dir, curr_epoch, 
                                  prefix="det_")
            utils.save_checkpoint(model_inp, optimizer_inp, global_step, 
                                  checkpoint_dir, curr_epoch, 
                                  prefix="inp_")

        global_step += 1
        
    return global_step


def run_training(
    train_data_loader, 
    model_det, optimizer_det, 
    model_inp, optimizer_inp, 
    checkpoint_dir, 
    global_step, start_epoch, total_epoches):
    
    loss_module = Loss(device)
    for curr_epoch in range(start_epoch, total_epoches):        
        global_step = run_epoch(
            train_data_loader,
            model_det, optimizer_det, 
            model_inp, optimizer_inp, 
            checkpoint_dir, 
            global_step, curr_epoch, 
            loss_module)


def run_main():
    global_step = 0
    global_epoch = 0
    batch_size = 8
    total_epoches = 36

    checkpoint_dir = "./checkpoints"
    checkpoint_det_path = None 
    checkpoint_inp_path = None
    
    ffhq_folder = "data/ffhq"
    ffhqr_folder = "data/ffhqr"

    model_det = DetectionUNet(n_channels=3, n_classes=1).to(device)
    optimizer_det = optim.AdamW(
        [p for p in model_det.parameters() if p.requires_grad],
        lr=initial_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
        eps=eps
    )
    
    model_inp = RetouchingNet(in_channels=4, out_channels=3).to(device)
    optimizer_inp = optim.AdamW(
        [p for p in model_inp.parameters() if p.requires_grad],
        lr=initial_learning_rate, 
        betas=(0.5, 0.999), 
        weight_decay=weight_decay, 
        eps=eps
    )

    if checkpoint_det_path:
        model_det, optimizer_det, global_step, global_epoch = utils.load_checkpoint(
                checkpoint_det_path, model_det, optimizer_det, 
                overwrite_global_states=True,
                use_cuda=("cuda" in device)
            )
    
    if checkpoint_inp_path:
        model_inp, optimizer_inp, global_step, global_epoch = utils.load_checkpoint(
                checkpoint_inp_path, model_inp, optimizer_inp, 
                overwrite_global_states=True,
                use_cuda=("cuda" in device)
            )

    train_dataset = JointDataset(ffhq_folder, ffhqr_folder)    
    
    start_epoch = global_step // (len(train_dataset)//batch_size)
    
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    
    run_training(train_data_loader, 
                 model_det, optimizer_det, 
                 model_inp, optimizer_inp, 
                 checkpoint_dir,
                 global_step, start_epoch, total_epoches)




if __name__ == "__main__":
    run_main()
    