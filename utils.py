# partly from https://github.com/modelscope/modelscope/blob/master/modelscope/models/cv/skin_retouching/utils.py

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'gen_diffuse_mask', 'get_crop_bbox', 'get_roi_without_padding',
    'patch_aggregation_overlap', 'patch_partition_overlap', 'preprocess_roi',
    'resize_on_long_side', 'roi_to_tensor', 'smooth_border_mg', 'whiten_img'
]


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(
            img, (int(src_width * scale), long_side),
            interpolation=cv2.INTER_LINEAR)
    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(
            img, (long_side, int(src_height * scale)),
            interpolation=cv2.INTER_LINEAR)

    return _img, scale


def get_crop_bbox(detecting_results):
    boxes = []
    for anno in detecting_results:
        if anno['score'] == -1:
            break
        boxes.append({
            'x1': anno['bbox'][0],
            'y1': anno['bbox'][1],
            'x2': anno['bbox'][2],
            'y2': anno['bbox'][3]
        })
    face_count = len(boxes)

    suitable_bboxes = []
    for i in range(face_count):
        face_bbox = boxes[i]

        face_bbox_width = abs(face_bbox['x2'] - face_bbox['x1'])
        face_bbox_height = abs(face_bbox['y2'] - face_bbox['y1'])

        face_bbox_center = ((face_bbox['x1'] + face_bbox['x2']) / 2,
                            (face_bbox['y1'] + face_bbox['y2']) / 2)

        square_bbox_length = face_bbox_height if face_bbox_height > face_bbox_width else face_bbox_width
        enlarge_ratio = 1.5
        square_bbox_length = int(enlarge_ratio * square_bbox_length)

        sideScale = 1

        square_bbox = {
            'x1':
            int(face_bbox_center[0] - sideScale * square_bbox_length / 2),
            'x2':
            int(face_bbox_center[0] + sideScale * square_bbox_length / 2),
            'y1':
            int(face_bbox_center[1] - sideScale * square_bbox_length / 2),
            'y2': int(face_bbox_center[1] + sideScale * square_bbox_length / 2)
        }

        suitable_bboxes.append(square_bbox)

    return suitable_bboxes


def get_roi_without_padding(img, bbox):
    crop_t = max(bbox['y1'], 0)
    crop_b = min(bbox['y2'], img.shape[0])
    crop_l = max(bbox['x1'], 0)
    crop_r = min(bbox['x2'], img.shape[1])
    roi = img[crop_t:crop_b, crop_l:crop_r]
    return roi, 0, [crop_t, crop_b, crop_l, crop_r]


def get_roi(img, bbox):
    roi, index, pos = get_roi_without_padding(img, bbox)
    return roi


def roi_to_tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))[None, ...]

    return img


def preprocess_roi(img):
    # img = img.float() / 255.0
    img = (img - 0.5) * 2

    return img

def postrocess_roi(img):
    # img = img.clamp(-1.0, 1.0)
    img = (img + 1.0) / 2
    return img


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result



def patch_partition_overlap(image, p1, p2, padding=32):

    B, C, H, W = image.size()
    h, w = H // p1, W // p2
    image = F.pad(
        image,
        pad=(padding, padding, padding, padding, 0, 0),
        mode='constant',
        value=0)

    patch_list = []
    for i in range(h):
        for j in range(w):
            patch = image[:, :, p1 * i:p1 * (i + 1) + padding * 2,
                          p2 * j:p2 * (j + 1) + padding * 2]
            patch_list.append(patch)

    output = torch.cat(
        patch_list, dim=0)  # (b h w) c (p1 + 2 * padding) (p2 + 2 * padding)
    return output


def patch_aggregation_overlap(image, h, w, padding=32):

    image = image[:, :, padding:-padding, padding:-padding]

    output = rearrange(image, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h=h, w=w)

    return output


def smooth_border_mg(diffuse_mask, mg):
    mg = mg - 0.5
    diffuse_mask = F.interpolate(
        diffuse_mask, mg.shape[:2], mode='bilinear')[0].permute(1, 2, 0)
    mg = mg * diffuse_mask
    mg = mg + 0.5
    return mg


def whiten_img(image, skin_mask, whitening_degree, flag_bigKernal=False):
    """
    image: rgb
    """
    dilate_kernalsize = 30
    if flag_bigKernal:
        dilate_kernalsize = 80
    new_kernel1 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernalsize, dilate_kernalsize))
    new_kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_kernalsize, dilate_kernalsize))
    if len(skin_mask.shape) == 3:
        skin_mask = skin_mask[:, :, -1]
    skin_mask = cv2.dilate(skin_mask, new_kernel1, 1)
    skin_mask = cv2.erode(skin_mask, new_kernel2, 1)
    skin_mask = cv2.blur(skin_mask, (20, 20)) / 255.0
    skin_mask = skin_mask.squeeze()
    skin_mask = torch.from_numpy(skin_mask).to(image.device)

    print(f"skin_mask: {skin_mask.shape}")
    skin_mask = torch.stack([skin_mask, skin_mask, skin_mask], dim=0)[None,
                                                                      ...]
    print(f"torch skin_mask: {skin_mask.size()}")
    
    skin_mask[:, 1:, :, :] *= 0.75

    whiten_mg = skin_mask * 0.2 * whitening_degree + 0.5
    assert len(whiten_mg.shape) == 4
    whiten_mg = F.interpolate(
        whiten_mg, image.shape[:2], mode='bilinear')[0].permute(1, 2, 0).half()

    print(f"whiten_mg: {whiten_mg.size()}, output_pred: {image.size()}")
    
    output_pred = image.half()
    output_pred = output_pred / 255.0
    output_pred = (
        -2 * whiten_mg + 1
    ) * output_pred * output_pred + 2 * whiten_mg * output_pred  # value: 0~1
    output_pred = output_pred * 255.0
    output_pred = output_pred.byte()

    output_pred = output_pred.cpu().numpy()
    return output_pred


def gen_diffuse_mask(out_channels=3):
    mask_size = 500
    diffuse_with = 20
    a = np.ones(shape=(mask_size, mask_size), dtype=np.float32)

    for i in range(mask_size):
        for j in range(mask_size):
            if i >= diffuse_with and i <= (
                    mask_size - diffuse_with) and j >= diffuse_with and j <= (
                        mask_size - diffuse_with):
                a[i, j] = 1.0
            elif i <= diffuse_with:
                a[i, j] = i * 1.0 / diffuse_with
            elif i > (mask_size - diffuse_with):
                a[i, j] = (mask_size - i) * 1.0 / diffuse_with

    for i in range(mask_size):
        for j in range(mask_size):
            if j <= diffuse_with:
                a[i, j] = min(a[i, j], j * 1.0 / diffuse_with)
            elif j > (mask_size - diffuse_with):
                a[i, j] = min(a[i, j], (mask_size - j) * 1.0 / diffuse_with)
    a = np.dstack([a] * out_channels)
    return a


def pad_to_size(
    target_size: Tuple[int, int],
    image: np.array,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, Tuple[int, int, int, int]]]:
    """Pads the image on the sides to the target_size

    Args:
        target_size: (target_height, target_width)
        image:
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns:
        {
            "image": padded_image,
            "pads": (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    target_height, target_width = target_size

    image_height, image_width = image.shape[:2]

    if target_width < image_width:
        raise ValueError(f'Target width should bigger than image_width'
                         f'We got {target_width} {image_width}')

    if target_height < image_height:
        raise ValueError(f'Target height should bigger than image_height'
                         f'We got {target_height} {image_height}')

    if image_height == target_height:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = target_height - image_height
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if image_width == target_width:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = target_width - image_width
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    result = {
        'pads': (x_min_pad, y_min_pad, x_max_pad, y_max_pad),
        'image':
        cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad,
                           cv2.BORDER_CONSTANT),
    }

    if bboxes is not None:
        bboxes[:, 0] += x_min_pad
        bboxes[:, 1] += y_min_pad
        bboxes[:, 2] += x_min_pad
        bboxes[:, 3] += y_min_pad

        result['bboxes'] = bboxes

    if keypoints is not None:
        keypoints[:, 0] += x_min_pad
        keypoints[:, 1] += y_min_pad

        result['keypoints'] = keypoints

    return result


def unpad_from_size(
    pads: Tuple[int, int, int, int],
    image: Optional[np.array] = None,
    bboxes: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Crops patch from the center so that sides are equal to pads.

    Args:
        image:
        pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
        bboxes: np.array with shape (num_boxes, 4). Each row: [x_min, y_min, x_max, y_max]
        keypoints: np.array with shape (num_keypoints, 2), each row: [x, y]

    Returns: cropped image

    {
            "image": cropped_image,
            "bboxes": shifted_boxes,
            "keypoints": shifted_keypoints
        }

    """
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads

    result = {}

    if image is not None:
        height, width = image.shape[:2]
        result['image'] = image[y_min_pad:height - y_max_pad,
                                x_min_pad:width - x_max_pad]

    if bboxes is not None:
        bboxes[:, 0] -= x_min_pad
        bboxes[:, 1] -= y_min_pad
        bboxes[:, 2] -= x_min_pad
        bboxes[:, 3] -= y_min_pad

        result['bboxes'] = bboxes

    if keypoints is not None:
        keypoints[:, 0] -= x_min_pad
        keypoints[:, 1] -= y_min_pad

        result['keypoints'] = keypoints

    return result


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    root = dir_path
    def _scandir(dir_path, suffix, recursive):
        # print(dir_path, os.scandir(dir_path))
        for entry in os.scandir(dir_path):
            # print(entry, not entry.name.startswith('.') and entry.is_file(), full_path)
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    # print(entry.path, root)
                    return_path = os.path.relpath(entry.path, root)
                    

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue
    return _scandir(dir_path, suffix=suffix, recursive=recursive)



def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder, recursive=True))
    paths = [os.path.join(folder, path) for path in paths]
    return paths



def calc_lr(initial_learning_rate, curr_epoch, total_epoches=200):
    lr = initial_learning_rate - initial_learning_rate * (1-0.1) * (curr_epoch)/total_epoches
    return lr


def adjust_lr(optimizer, initial_learning_rate, curr_epoch, total_epoches=200):
    new_lr = calc_lr(initial_learning_rate, curr_epoch, total_epoches)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    print(f"optimizer learning rate changed to {new_lr}")


def _load(checkpoint_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_weights(path, model, use_cuda=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda=use_cuda)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    return model


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True, use_cuda=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda=use_cuda)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model, optimizer, global_step, global_epoch


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = os.path.join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
