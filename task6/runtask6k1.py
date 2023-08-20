import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2
from mmengine import Config
cfg = Config.fromfile('Zihao-Configs/ZihaoDataset_Segformer_20230818.py')
checkpoint_path = 'work_dirs/Zihao_Segformer.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')
img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/val/01bd15599c606aa801201794e1fa30.jpg'
img_bgr = cv2.imread(img_path)
plt.figure(figsize=(8, 8))
plt.imshow(img_bgr[:,:,::-1])
plt.savefig('outputs/k1.jpg')
plt.show()
result = inference_model(model, img_bgr)
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
np.unique(pred_mask)
plt.figure(figsize=(8, 8))
plt.imshow(pred_mask)
plt.savefig('outputs/K1-0.jpg')
plt.show()