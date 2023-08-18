import os
# os.chdir('mmsegmentation')
print(os.getcwd())
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib inline
img_path = 'Watermelon87_Semantic_Seg_Mask/img_dir/train/045_sozai_l.jpg'
mask_path = 'Watermelon87_Semantic_Seg_Mask/ann_dir/train/045_sozai_l.png'
img = cv2.imread(img_path)
mask = cv2.imread(mask_path)
print(img.shape)
print(mask.shape)
np.unique(mask)
palette = [
    ['background', [127,127,127]],
    ['red', [0,0,200]],
    ['green', [0,200,0]],
    ['white', [144,238,144]],
    ['seed-black', [30,30,30]],
    ['seed-white', [8,189,251]]
]
palette_dict = {}
for idx, each in enumerate(palette):
    palette_dict[idx] = each[1]
print("palette_dict")   
print(palette_dict) 
mask = mask[:,:,0]

# 将预测的整数ID，映射为对应类别的颜色
viz_mask_bgr = np.zeros((mask.shape[0], mask.shape[1], 3))
for idx in palette_dict.keys():
    viz_mask_bgr[np.where(mask==idx)] = palette_dict[idx]
viz_mask_bgr = viz_mask_bgr.astype('uint8')

# 将语义分割标注图和原图叠加显示
opacity = 0.2 # 透明度越大，可视化效果越接近原图
label_viz = cv2.addWeighted(img, opacity, viz_mask_bgr, 1-opacity, 0)
# cv2.__version__
print(cv2.__version__) 
plt.imshow(label_viz[:,:,::-1])
# plt.imsave('outputs/045_sozai_l.png', label_viz[:,:,::-1])
plt.show()
cv2.imwrite('outputs/D-1.jpg', label_viz)