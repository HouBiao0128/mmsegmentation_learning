import matplotlib 
import matplotlib.pyplot as plt
print(matplotlib.__file__)
# wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf -O /home/joseph/anaconda3/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf

matplotlib.rc("font",family='SimHei') # 中文字体
plt.plot([1,2,3], [100,500,300])
plt.title('matplotlib中文字体测试', fontsize=25)
plt.xlabel('X轴', fontsize=15)
plt.ylabel('Y轴', fontsize=15)
plt.show()