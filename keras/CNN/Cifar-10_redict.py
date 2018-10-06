
import numpy as np
import scipy.misc # 以图像形式保存数组
from keras.models import model_from_json
from keras.optimizers import SGD

# 加载模型
model_architerture = 'cifar10_architerture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architerture).read())
model.load_weights(model_weights)

# 加载图片
img_names = ['cat-standing.jpg', 'dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (1, 0, 2)).astype('float')
        for img_name in img_names]
imgs = np.array(imgs) / 255

# 训练
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

# 预测
predictions = model.predict_classes(imgs)
print(predictions)