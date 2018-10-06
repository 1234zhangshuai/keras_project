# 首先载入MNIST数据集，并创建默认的 Interactive Session
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 给权重制造一些噪声（如截断的正态分布噪声），将标准差设为0.1。定义变量，初始化为截断正太分布的变量
def weight_variable(shape):#定义权重
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 因为要使用ReLU,给偏置增加一些小的正值0.1，用来避免死亡节点。定义变量，初始化为常量
def bias_variable(shape):#定义偏置
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # 定义一个二维卷积函数，x是输入,w是卷积的参数，核函数
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # 中间两个1表示x方向和y方向的步长

def max_pool_2x2(x): # 定义一个最大池化函数
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784]) # x是特征,转换为28*28的图像
y_ = tf.placeholder(tf.float32, [None, 10]) # y_是真实的label，0-9个手写数字作为标签
x_image = tf.reshape(x, [-1, 28, 28, 1]) # -1代表样本数量不固定,28*28*1

# 第一个卷积层
W_conv1 = weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32]) #
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) # 进行第一层卷积，并使用激活函数进行非线性处理，28*28*32
h_pool1 = max_pool_2x2(h_conv1) # 使用最大池化函数对卷积的输出结果进行池化操作，14*14*32

# 第二个卷积层
W_conv2 = weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2) # 14*14*64
h_pool2 = max_pool_2x2(h_conv2) # 7*7*64

# 一个全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# 为了减轻过拟合，使用了一个dropout层，通过一个placeholder传入keep_prob比率来控制
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 将dropout层的输出连接一个softmax层，得到最后的概率输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matful(h_fc1_drop, W_fc2)+b_fc2)

# 定义损失函数为cross_entropy;优化器使用Adam，并给予一个比较小的学习速率1e-4
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义评测准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(10):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
