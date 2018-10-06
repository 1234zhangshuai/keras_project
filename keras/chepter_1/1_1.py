
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Dense(12, input_dim=8, kernel_initializer='random_uniform'))