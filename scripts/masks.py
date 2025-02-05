import os
import cv2
import pickle
import numpy as np
from PIL import Image
from typing import Any, Dict
from modelscope.pipelines.base import Input, Pipeline
from modelscope.preprocessors import LoadImage
import torch
import onnxruntime

import utils

def load_onnx_model(onnx_path):
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
        print("cuda...")
    sess = onnxruntime.InferenceSession(onnx_path,providers=providers)
    out_node_name = []
    input_node_name = []
    for node in sess.get_outputs():
        out_node_name.append(node.name)
    for node in sess.get_inputs():
        input_node_name.append(node.name)
    return sess, input_node_name, out_node_name

patch_size = 512
skin_model_path = "./third_party/skin/model.onnx"
sess, input_node_name, out_node_name = load_onnx_model(skin_model_path)


ffhq_folder = "data/ffhq"
ffhq_paths = utils.paths_from_folder(ffhq_folder)
print(f"ffhq_imgs: {len(ffhq_paths)}")

for i, ffhq_path in enumerate(ffhq_paths):
    basename = os.path.basename(ffhq_path)
    dst_path = os.path.join("data/skins", basename)
    if os.path.exists(dst_path) == True:
        continue
    if i % 100 == 0:
        print(i)
    ffhq_img = Image.open(ffhq_path)
    im = LoadImage.convert_to_ndarray(ffhq_img)
    rgb_image = im.astype(np.uint8)
    rgb_image_small = rgb_image.copy()
    input_feed = {}
    input_feed[input_node_name[0]] = rgb_image_small.astype('float32')
    skin_mask = sess.run(out_node_name, input_feed=input_feed)[0]
    cv2.imwrite(dst_path, skin_mask.copy())
