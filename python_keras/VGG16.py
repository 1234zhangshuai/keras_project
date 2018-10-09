
from keras import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Input
from keras.optimizers import SGD

model = Sequential()

# BLOCK 1
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1',
                 input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool'))

# BLOCK2
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool'))

# BLOCK3
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool'))

# BLOCK4
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool'))

# BLOCK5
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten())
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax', name='prediction'))

model.summary()
