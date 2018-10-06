# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:00:19 2018

@author: Administrator
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
sess=tf.InteractiveSession()
#给权值制造一些随机高斯噪声来打破完全对称，同时因为激活函数使用ReLU,也给偏置增加一些小的正值（0.1）来避免死亡节点。
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#接下来定义卷积层，下面x代表输入，W是卷积的参数如[5,5,1,32]代表卷积核大小5x5，1通道，数目为32，strides都为1代表划过图片每一个点
#padding='SAME'代表加入的padding使得卷积输入输出保持同样的尺寸。
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#接下来定义池化层，这里使用将2x2的最大池化降为1x1的像素
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#定义输入的placeholder，x是特征，y_是真实标签，因为卷积会利用空间信息，需要将1D输入向量转为2D图片结构，1x784->28x28，下面的-1代表样本数量不固定，最后的1代表颜色通道数
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
#定义第一个卷积层，先卷积再加上偏置，在使用ReLU激活函数进行非线性处理，最后进行池化操作
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#定义第二层卷积层，卷积核数量变为64
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#将第二个卷积层的输出转成1D的向量，然后连接一个全连接层，隐含节点为1024，并使用ReLU激活函数
W_fc1=weight_variable([7 * 7 * 64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7 * 7 * 64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
#为了减轻过拟合，下面使用一个Dropout层，通过一个placeholder传入keep_prob比率来控制的。
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#最后将Dropout层的输出连接一个Softmax层，得到最后的概率输出
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
#定义损失函数为交叉熵，优化器使用Adam，并给予一个比较小的学习速率1e-4
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#再继续定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#下面开始训练过程
tf.global_variables_initializer().run()
for i in range(20):
    batch = mnist.train.next_batch(50)
    if i%100 ==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
#在最终的测试集上进行全面的测试，得到整体的分类准确率
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))