import onnxruntime
import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
# %matplotlib inline
onnx_path = 'mmseg2onnx_fastscnn/end2end.onnx'
ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
x = torch.randn(1, 3, 1024, 2048).numpy()
ort_inputs = {'input': x}

# onnx runtime 输出
ort_output = ort_session.run(['output'], ort_inputs)[0]
img_path = 'watermelon_test1.jpg'
img_bgr = cv2.imread(img_path)
plt.imshow(img_bgr[:,:,::-1])
plt.show()
# 获取原图宽高
h, w = img_bgr.shape[0], img_bgr.shape[1]

# 原图中心点坐标
center_x, center_y = w // 2, h // 2
ratio = 2 # 设置裁剪尺寸缩放系数（1024的倍数，不要超过原图范围）
new_h = 1024 * ratio
new_w = 2048 * ratio

img_bgr_crop = img_bgr[int(center_y-new_h/2):int(center_y+new_h/2), int(center_x-new_w/2):int(center_x+new_w/2)]
plt.imshow(img_bgr_crop[:,:,::-1])
plt.show()
img_bgr_resize = cv2.resize(img_bgr_crop, (2048, 1024)) # 缩放尺寸
plt.imshow(img_bgr_resize[:,:,::-1])
plt.show()
img_tensor = img_bgr_resize

# BGR 三通道的均值
mean = (123.675, 116.28, 103.53)

# BGR 三通道的标准差
std = (58.395, 57.12, 57.375)

# 归一化
img_tensor = (img_tensor - mean) / std
img_tensor = img_tensor.astype('float32')

# BGR 转 RGB
img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2RGB)

# 调整维度
img_tensor = np.transpose(img_tensor, (2, 0, 1))
# 扩充 batch-size 维度
input_tensor = np.expand_dims(img_tensor, axis=0)
# ONNX Runtime 输入
ort_inputs = {'input': input_tensor}
# onnx runtime 输出
ort_output = ort_session.run(['output'], ort_inputs)[0]
pred_mask = ort_output[0][0]
np.unique(pred_mask)
plt.imshow(pred_mask)
plt.show()
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
opacity = 0.3 # 透明度，越大越接近原图
# 将预测的整数ID，映射为对应类别的颜色
pred_mask_bgr = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))
for idx in palette_dict.keys():
    pred_mask_bgr[np.where(pred_mask==idx)] = palette_dict[idx]
pred_mask_bgr = pred_mask_bgr.astype('uint8')

# 将语义分割预测图和原图叠加显示
pred_viz = cv2.addWeighted(img_bgr_resize, opacity, pred_mask_bgr, 1-opacity, 0)
plt.imshow(pred_viz[:,:,::-1])
plt.show()
plt.imshow(img_bgr_resize[:,:,::-1])
plt.show()
cv2.imwrite('N2.jpg', pred_viz)