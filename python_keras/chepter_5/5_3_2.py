
from keras import models
from keras import layers
from keras.applications import VGG16

# 将VGG16卷积实例化
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 在卷积上添加一个密集连接分类器
model = models.Sequential()
model.add(conv_base)