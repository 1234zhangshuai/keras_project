
from keras.datasets import imdb
from Vec_seq import vectorize_sequences
from keras import models
from keras import layers
# from keras import optimizers
# from keras import losses
# from keras import metrics
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data) # 将训练数据向量化
x_test = vectorize_sequences(test_data) # 测试数据向量化

# 将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# # 模型定义
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
#
# # 留出验证集
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
#
# # 编译模型
# # 编译模型&配置优化器
# # model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy
# #               , metrics=[metrics.binary_accuracy])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# # 训练模型
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#
# # 绘制训练损失和验证损失
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, len(loss_values) + 1)
#
# plt.plot(epochs, loss_values, 'bo', label='Training loss') # bo表示蓝色圆点
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss') # b表示蓝色圆点
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# # 绘制训练精度和验证精度
# plt.clf() # 清空图像
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()

# 从头开始重新训练一个模型
# 模型定义
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 编译模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
# print(model.predict(x_test))