
from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0) # 数据平均值
train_data -= mean
std = train_data.std(axis=0) # 标准差
train_data /= std

test_data -= mean
test_data /= std

# 模型定义
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# 保存每折的验证结果
k = 4
num_val_sample = len(train_data) // k # 浮点除法，对结果进行四舍五入
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_sample: (i + 1 * num_val_sample)] # 准备验证数据：第k个分区的数据
    val_targets = train_targets[i * num_val_sample: (i + 1) * num_val_sample]

    # 准备训练数据：其他所有分区的数据
    partial_train_data = np.concatenate([train_data[:i * num_val_sample],
                                         train_data[(i + 1) * num_val_sample:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_sample],
                                            train_targets[(i + 1) * num_val_sample:]], axis=0)

#     model = build_model() # 构建Kreas模型（已编译）
#     # 训练模式（静默模式，verbose=0）
#     history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
#                         epochs=num_epochs, batch_size=1, verbose=0)
#     # val_me, val_mae = model.evaluate(val_data, val_targets, verbose=0) # 在验证数据上评估模型
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)
#
# # 计算所有轮次中的K折验证分数平均值
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return  smoothed_points
#
#
# smooth_mae_history = smooth_curve(average_mae_history[10:])
#
# # 绘制验证分数(删除前10个数据点)
# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

# 训练最终模型
# 构建Kreas模型（已编译）
model = build_model()
# 训练模式（静默模式，verbose=0）在所有训练数据上训练模型
history = model.fit(train_data, train_targets, epochs=num_epochs, batch_size=16, verbose=0)
# 在验证数据上评估模型
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
