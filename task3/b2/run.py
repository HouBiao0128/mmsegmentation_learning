# import os
# os.chdir('mmsegmentation')
import numpy as np
import cv2

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv

import matplotlib.pyplot as plt
# %matplotlib inline
# 模型 config 配置文件
config_file = 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py'

# 模型 checkpoint 权重文件
checkpoint_file = 'checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')
img_path = 'data/street_uk.jpeg'


img_bgr = cv2.imread(img_path)

result = inference_model(model, img_bgr)
pred_mask = result.pred_sem_seg.data[0].detach().cpu().numpy()
plt.imsave('outputs/B1_uk_segformer2.png', pred_mask)
# plt.imshow(pred_mask)
# plt.show()

