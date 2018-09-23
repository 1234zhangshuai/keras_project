
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image # 图像预处理工具的模块
import os
import matplotlib.pyplot as plt

# original_dataset_dir = 'F:/python/Deeplearn_keras/kaggle_original_data/train'

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
# 将猫狗分类的小型卷积神经网络实例化
# 卷积与迟化
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
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#############################################################################
# 配置模型用于训练
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
#############################################################################

# 数据预处理步骤
# 1、读取图像数据
# 2、将JPEG文件解码为RGB像素网络
# 3、将这些像素网络转换为浮点数张量
# 4、将像素值（0~255范围内）缩放到[0, 1]区间（神经网络喜欢处理较小的输入值）

# 使用ImageDataGenerator从目录中读取图像
train_datagen = ImageDataGenerator(rescale=1./255) # 将所有图像乘以1/255缩放
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, # 目标目录
    target_size=(150, 150), # 将所有图像的大小调整为150 x 150
    batch_size=20, class_mode='binary') # 因为使用了binary_crossentropy损失，所以需要用二进制标签

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20, class_mode='binary')

#############################################################################
# 利用批量生成器拟合模型
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
# 保存模型
model.save('cats_and_dogs_small_1.h5')
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
