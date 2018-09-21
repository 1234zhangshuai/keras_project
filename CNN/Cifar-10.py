
from keras.preprocessing.image import ImageDataGenerator #通过调整、水平/垂直翻转、缩放、信道交换的多类型的转换来增加图形
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

NUM_TO_AUGMENT = 5
# CIFAR-10是一个包含了60 000张32 x 32像素的三通道图像数据集
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# 常量
BATH_SIZE = 128
NB_CLASSES = 10
NB_EPOCH = 20
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# 加载数据集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 扩展
print("Augmenting training set image...")
dategen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

xtas, ytas = [], []
for i in range(X_train.shape[0]):
    num_aug = 0
    x = X_train[i] # (3, 32, 32)
    x = x.reshape((1,) + x.shape) # (1, 3, 32, 32)
    for x_aug in dategen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cifar', save_format='jpeg'):
        if num_aug >= NUM_TO_AUGMENT:
            break
        xtas.append(x_aug[0])
        num_aug += 1

# 做one-hot编码, 并把图像归一化
# 分类转换
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 看成float类型并归一化
X_train = X_train.astype('float')
X_test = X_test.astype('float')
X_train /= 255
X_test /= 255

# 网络&改进
# conv + conv + maxpool + dropout + conv + conv + maxpool, 3(2 + 1) + 1 + 3(2 + 1)
# 其后是标准的 dense + dropout + dense, 所有激活函数都是relu
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# 全连接
model.add(Flatten())
# dense + dropout + dense
model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# 匹配数据
dategen.fit(X_train)

# 训练
model.compile(loss='categorical_crossentropy', optimizer=OPTIM, metrics=['accuracy'])
# history = model.fit(X_train, Y_train, epochs=NB_EPOCH, batch_size=BATH_SIZE,
#                     verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
history = model.fit_generator(dategen.flow(X_train, Y_train, batch_size=BATH_SIZE),
                              samples_per_epoch=X_train.shape[0], epochs=NB_EPOCH, verbose=VERBOSE)

# 测试评估
score = model.evaluate(X_test, Y_test, batch_size=BATH_SIZE, verbose=VERBOSE)

print("Test score:", score[0])
print("Test accuracy:", score[1])

# 保存模型
model_json = model.to_json()
open('cifar10_architecture.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)

# 列出全部历史数据
print(history.history.keys())
# 汇总准确率的历史数据
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 汇总损失历史数据
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()