# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
import scipy
import os
import csv
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import normalize


# 创建placeholders对象
def create_placeholders(n_x, n_y):
    """
    placeholder是TensorFlow的占位符节点，由placeholder方法创建，其也是一种常量，但是由用户在调用run方法是传递的.
    也可以将placeholder理解为一种形参。
    即其不像constant那样直接可以使用，需要用户传递常数值。
    """
    X = tf.placeholder(tf.float32, shape=[None, n_x], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, n_y], name='Y')

    return X, Y


# 初始化参数
def initialize_parameters(n):
    tf.set_random_seed(1)
    Z0 = tf.get_variable("Z0", shape=[1, 2 * n],initializer=tf.contrib.layers.xavier_initializer(seed=1))  # 初始化X_feature和X_knockoffs的的系数
    Zb0 = tf.get_variable("Zb0", shape=[1, 2 * n], initializer=tf.zeros_initializer())
    W0 = tf.get_variable("W0", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b0 = tf.get_variable("b0", shape=[1, n], initializer=tf.zeros_initializer())
    W1 = tf.get_variable("W1", shape=[n, n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", shape=[n, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", shape=[n, n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", shape=[n, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", shape=[2, n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", shape=[2, 1], initializer=tf.zeros_initializer())

    parameters = {"Z0": Z0,
                  "Zb0": Zb0,
                  "W0": W0,
                  "b0": b0,
                  "W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  }
    return parameters


# 前向传播
def forward_propagation(X, parameters, n, lambd):
    print(type(n), n)
    Z0 = parameters['Z0']
    Zb0 = parameters['Zb0']
    W0 = parameters['W0']
    b0 = parameters['b0']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # 正则化
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W1))
    # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W3))

    X_temp = tf.multiply(Z0, tf.transpose(X))
    X0 = X_temp[:, 0:n]
    X1 = X_temp[:, n:2 * n]
    X0 = tf.add(X0, X1)

    # x0*w0+x0_knockoffs*w0'+x1*w1+x1_knockofss*w1'....
    A0 = tf.add(tf.multiply(W0, X0), b0)
    A0 = tf.transpose(A0)

    Z1 = tf.add(tf.matmul(W1, A0), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)

    # Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    # A2=tf.nn.relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A1), b3)
    print('Z3.shape: ',Z3.shape)
    return Z3


def compute_cost(Z3, Y):
    print(Z3.shape)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'))


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01, minibatch_size=10, num_epochs=30000, print_cost=True):
    tf.set_random_seed(1)
    (m, n_x) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    # 创建Placeholders,一个张量
    X, Y = create_placeholders(n_x, n_y)
    print(X.shape, Y.shape)
    # 初始化参数
    parameters = initialize_parameters(int(n_x / 2))
    # 前向传播
    Z3 = forward_propagation(X, parameters,  int(n_x / 2),0.002)
    # 计算代价
    cost = compute_cost(Z3, Y)

    # 后向传播: 定义tensorflow optimizer对象，这里使用AdamOptimizer.
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # 初始化所有参数
    init = tf.global_variables_initializer()

    # 启动session来计算tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            test_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
            epoch_cost = epoch_cost[1]

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("test_cost: ", test_cost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        # 神经网络经过训练后得到的值

        correct_prediction = tf.equal(tf.argmax(Z3, axis=1), tf.argmax(Y, 1))  # tf.argmax找出每一列最大值的索引
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # tf.cast转化数据类型
        print("Train Accuracy:", sess.run(accuracy, feed_dict={X: X_train, Y: Y_train}))
        print("Test Accuracy: ", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))

        return parameters


def loadDataSet(frx):
    Data=[];
    X = [];
    Y = [];
    name = []

    # 打开保存特征X的文件
    frx = open(frx)
    # 是否跳过第一行
    lines = frx.readlines()
    # 基因的名字
    name = lines[0].strip().split('\t')
    for line in range(1, len(lines)):
        curLine = lines[line].strip().split('\t')
        # 字符型转化为浮点型
        fltLine = list(map(float, curLine))
        Data.append(fltLine)

    # 转化为矩阵
    Data = np.mat(Data)
    X=Data[:,:-1]
    Y=Data[:,-1]
    m, n = X.shape

    # 划分训练集和测试集
    indices = np.arange(m)
    # random_state表示每次生成的训练集和测试集都是固定的，也就是结果可以重复
    X_train, X_test, Y_train, Y_test, index1, index2 = train_test_split(X, Y, indices, test_size=0.2, random_state=42)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test, name, index1

def analyse(parameters, n):
    print(type(n))
    Z0 = parameters['Z0']
    W0 = parameters['W0']
    b0 = parameters['b0']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']

    # 进行特征重要性计算
    # W = np.matmul(W3, W2)
    W = np.matmul(W3, W1)
    # W=W3

    Z1 = np.multiply(Z0[:, 0:n], W0)
    Z2 = np.multiply(Z0[:, n:2 * n], W0)

    Z11 = np.multiply(Z1, W)
    Z22 = np.multiply(Z2, W)
    # print(Z0)
    res = {}  # 公式为W0*Z0
    res1 = {}  # Z0[i]-Z0[j],
    res2 = {}

    for i in range(0, n):
        res[i] = Z1[0][i] * Z1[0][i] - Z2[0][i] * Z2[0][i]

    for i in range(0, n):
        res1[i] = Z0[0][i] * Z0[0][i] - Z0[0][n + i] * Z0[0][n + i]

    for i in range(0, n):
        res2[i] = Z11[0][i] * Z11[0][i] - Z22[0][i] * Z22[0][i]
    # 不使用knockoffs时的结果
    # for i in range(0,215):
    #    res2[i]=W0[0][i]

    res = sorted(res.items(), key=lambda d: d[1], reverse=True)
    res1 = sorted(res1.items(), key=lambda d: d[1], reverse=True)
    res2 = sorted(res2.items(), key=lambda d: d[1], reverse=True)

    # (Z0-Z0')^2*W0
    """
     rank = 1
    for key in res:
        #print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or
                name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1
    """

    """
    rank = 1
    for key in res1:
        #print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or
                name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1
    """

    fr = open('./result.txt', 'a')
    res = []
    rank = 1
    for key in res2:
        print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])

        fr.write(str(rank) + '    ' + str(name[key[0]]) + '   ' + str(key[1]) + '\n')
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or
                name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1

if __name__ == '__main__':
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig, name, index1 = loadDataSet('./GY-DATA/1-001.txt')
    # 生成X对应的knockoffX数据
    knockoffX = []
    fr = open("./GY-DATA/1-001-knockoffs.txt")
    rank = 0
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        rank = rank + 1
        fltLine = list(map(float, curLine))
        knockoffX.append(fltLine)
    knockoffX = np.mat(knockoffX)
    # 只取训练集那一部分
    knockoffX = knockoffX[index1, :]

    print('knockoffX.shape: ', knockoffX.shape)

    X_train_orig = np.hstack((X_train_orig, knockoffX))
    X_train=normalize(X_train_orig)
    X_test=normalize(X_test_orig)
    Y_train = to_categorical(Y_train_orig)
    Y_test = to_categorical(Y_test_orig)
    parmeters = model(X_train, Y_train, X_test, Y_test)

