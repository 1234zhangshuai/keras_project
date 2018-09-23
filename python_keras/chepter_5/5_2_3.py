
from keras import  layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image # 图像预处理工具的模块
import os
import matplotlib.pyplot as plt

base_dir = 'F:/python/Deeplearn_keras/cats_and_dogs_small' # 保存(创建)较小数据集的目录
############################################################################
# 训练集目录
train_dir = os.path.join(base_dir, 'train')
# 验证集目录
validation_dir = os.path.join(base_dir, 'validation')
# 测试集目录
test_dir = os.path.join(base_dir, 'test')

# 猫的训练图像目录
train_cats_dir = os.path.join(train_dir, 'cats')
# 狗的训练图像目录
train_dogs_dir = os.path.join(train_dir, 'dogs')

# 猫的验证图像目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
# 狗的验证图像目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 猫的测试图像目录
test_cats_dir = os.path.join(test_dir, 'cats')
# 狗的测试图像目录
test_dogs_dir = os.path.join(test_dir, 'dogs')

#############################################################################
# print('total training cat images:', len(os.listdir(train_cats_dir))

# 将猫狗分类的小型卷积神经网络实例化
# 由于训练样本数据较少，所以将采用数据增强(原理：对图像执行多次随机变换)来降低过拟合问题

# # 利用ImageDataGenerator来设置数据增强
# datagen = ImageDataGenerator(rotation_range=40,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              shear_range=0.2, # 随机错切变换的角度
#                              zoom_range=0.2, # 图像随机缩放的范围
#                              horizontal_flip=True, # 随机将一半图像水平翻转
#                              fill_mode='nearest') # 用于填充新创建像素的方法
#
# # 显示几个随机增强后的训练图像
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
#
# img_path = fnames[3] # 选择一张图像进行增强
# img = image.load_img(img_path, target_size=(150, 150)) # 读取图像并调整大小
#
# x = image.img_to_array(img) # 将其装换为形状（150， 150， 3）的Numpy数组
# x = x.reshape((1,) + x.shape) # 将其形状改变为（1, 150, 150, 3）
#
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
#
# plt.show()
#############################################################################
# 定义一个包含dropout的新卷积神经网络
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# 全连接
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

#############################################################################
# 利用数据增强生成器训练卷积神经网络
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # 注意，不能增强验证数据

# 使用ImageDataGenerator从目录中读取图像
train_generator = train_datagen.flow_from_directory(
    train_dir, # 目标目录
    target_size=(150, 150), # 将所有图像的大小调整为150 x 150
    batch_size=32, class_mode='binary') # 因为使用了binary_crossentropy损失，所以需要用二进制标签

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32, class_mode='binary')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=validation_generator, validation_steps=50)
# 保存模型
model.save('cats_and_dogs_small_2.h5')
#############################################################################
# 绘制训练过程中的损失曲线和精度曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Train acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()

plt.show()
