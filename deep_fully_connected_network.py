# -*- coding:utf-8 -*-
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import load


# 导入load.py的数据
train_samples, train_labels = load._train_samples, load._train_labels
test_samples, test_labels = load._test_samples, load._test_labels

# print('Training set', train_samples.shape, train_labels.shape)
# print('    Test set', test_samples.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels

def get_chunk(samples, labels, chunkSize):
    '''
	Iterator/Generator: get a batch of data
	这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
	用于 for loop， just like range() function
    '''
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd


# 定义一个深度全连接网络的类
class deep_fully_connected_network():
    def __init__(self, num_hidden, batch_size):
        '''
        num_hidden:隐藏层节点数量
        batch_size:节省内存，所以分批处理数据。即每一批的数据量
        '''
        self.batch_size = batch_size
        self.test_batch_size = 500
        # 超参数
        self.num_hidden = num_hidden

        # Graph related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None

    def define_graph(self):
        '''
        定义计算图谱
            '''
        with self.graph.as_default():
            # 定义图谱中的各种变量
            self.tf_train_samples = tf.placeholder(
                tf.float32, shape=(self.batch_size, image_size, image_size, num_channels)
            )
            self.tf_train_labels = tf.placeholder(
                tf.float32, shape=(self.batch_size, num_labels)
            )
            self.tf_test_samples = tf.placeholder(
                tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels)
            )

            # 全连接层1 -->隐藏层
            fully_connected_layer1_weights = tf.Variable(
                tf.truncated_normal([image_size * image_size, self.num_hidden], stddev=0.1)
            )
            fully_connected_layer1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

            # 全连接层2 -->输出层
            fully_connected_layer2_weights = tf.Variable(
                tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1)
            )
            fully_connected_layer2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

            # 定义图谱的运算
            def model(data):
                # 全连接层1
                shape = data.get_shape().as_list()
                reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, fully_connected_layer1_weights) + fully_connected_layer1_biases)

                # 全连接层2
                return tf.matmul(hidden, fully_connected_layer2_weights) + fully_connected_layer2_biases

            # 训练计算 logits为全连接最后一个数的计算
            logits = model(self.tf_train_samples)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.tf_train_labels)
            )

            # 优化器
            self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

            # 预测
            self.train_prediction = tf.nn.softmax(logits)
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples))


    def run(self):
        '''
        用于Session
        '''
        self.session = tf.Session(graph=self.graph)
        with self.session as session:
            session.run(tf.global_variables_initializer())
            print('Start Training')
            # batch_size
            for i, samples, labels in get_chunk(train_samples, train_labels, chunkSize=self.batch_size):
                _, l, predictions = session.run(
                    [self.optimizer, self.loss, self.train_prediction],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )
                # labels is Ture Labels
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)


    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        '''
        计算预测的正确率和召回率
        '''
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm


if __name__ == '__main__':
    net = deep_fully_connected_network(num_hidden=128, batch_size=1000)
    net.define_graph()
    net.run()
