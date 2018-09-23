
from keras.datasets import reuters
from To_one_hot import to_one_hot
from Vec_seq import vectorize_sequences
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# print(len(train_data))
# print(len(test_data))
x_train = vectorize_sequences(train_data) # 将训练数据向量化
x_test = vectorize_sequences(test_data) # 测试数据向量化
# 将训练标签向量化
one_hot_train_labels = to_one_hot(train_labels)
# 将测试标签向量化
one_hot_test_labels = to_one_hot(test_labels)
# 获用keras内置函数如下
# from keras.utils.np_utils import to_categorical
#
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# # 模型定义
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
#
# # 编译模型
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
#
# # 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
#
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
#
# # 训练模型
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#
# # 绘制训练损失和验证损失
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# # 绘制训练精度和验证精度
# plt.clf() # 清空图像
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and bvalidation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()

# 从头开始重新训练一个模型
# 模型定义
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
# 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# 在新数据集上生成预测结果
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))