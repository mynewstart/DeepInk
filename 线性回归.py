# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
import scipy
import os

def loadDataSet(frx,fry):
    X=[];Y=[];name=[]

    #打开保存特征X的文件
    frx=open(frx)
    #是否跳过第一行
    lines=frx.readlines()
    #基因的名字
    name=lines[0].strip().split('\t')
    for line in range(1,len(lines) ):
            curLine=lines[line].strip().split('\t')
            #字符型转化为浮点型
            fltLine=list(map(float,curLine))
            X.append(fltLine)

    #转化为矩阵
    X=np.mat(X)
    #X=X[:,:19]
    m,n=X.shape

    #打开保存类别Y的文件
    fry=open(fry)
    for line in fry.readlines():
            curLine=line.strip().split('\t')
            fltLine=list(map(float,curLine))
            Y.append(fltLine)
    Y=np.mat(Y)
    #划分训练集和测试集
    indices=np.arange(m)
    #random_state表示每次生成的训练集和测试集都是固定的，也就是结果可以重复
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y,test_size=0.1,random_state=42)
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)  #维度分别是(1386,40),(1386,1),(595,40),(595,1)

    return X_train, X_test, Y_train, Y_test,name

#创建placeholders对象
def create_placeholders(n_x,n_y):
    """
    placeholder是TensorFlow的占位符节点，由placeholder方法创建，其也是一种常量，但是由用户在调用run方法是传递的.
    也可以将placeholder理解为一种形参。
    即其不像constant那样直接可以使用，需要用户传递常数值。
    """
    X=tf.placeholder(tf.float32,shape=[n_x,None],name="X")
    Y=tf.placeholder(tf.float32,shape=[n_y,None],name="Y")

    return X,Y

#初始化参数
def initialize_parameters(n,m):
    tf.set_random_seed(1)

    W = tf.get_variable("W", shape=[n,1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b = tf.get_variable("b", shape=[n, 1], initializer=tf.zeros_initializer())


    parameters = {
                  "W": W,
                  "b": b,
    }
    return parameters

#前向传播
def forward_propagation(X,parameters,lambd):
    W = parameters['W']
    b = parameters['b']
    #正则化
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(W))
    #tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(W2))
    #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W3))
    #tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(W4))

    Z3 = tf.add(tf.multiply(X , W), b)  # Z3 = np.dot(W3,Z2) + b3
    Z3=tf.reduce_sum(Z3,axis=0)
    print('Z3.shape: ',Z3.shape)

    return tf.transpose(Z3)
def compute_cost(Z3, Y):
    n_samples=Z3.shape[0]
    cost = tf.reduce_mean(tf.square(Z3 - Y))
    tf.add_to_collection('losses', cost)
    cost = tf.add_n(tf.get_collection('losses'))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          minibatch_size=10, num_epochs=20000, print_cost=True):

    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    print('shape: ',n_x,m)
    n_y = Y_train.shape[0]
    costs = []
    # 创建Placeholders,一个张量
    X, Y = create_placeholders(n_x, n_y)
    """下面的计算只是定义了一种计算的形式，并没有具体的数字，在实际run中使用这些函数来进行计算"""
    # 初始化参数
    parameters = initialize_parameters(n_x,m)
    # 前向传播
    Z3 = forward_propagation(X, parameters, 0.004)
    # 计算代价
    cost = compute_cost(Z3, Y)

    # 后向传播: 定义tensorflow optimizer对象，这里使用AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # 初始化所有参数
    init = tf.global_variables_initializer()

    # 启动session来计算tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            # 进行批度训练
            epoch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            test_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
            # print(epoch_cost)
            epoch_cost = epoch_cost[1]

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("test_cost: ", test_cost)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        # 神经网络经过训练后得到的值
        Z3 = sess.run(Z3, feed_dict={X: X_train, Y: Y_train})
        print(sess.run(cost, feed_dict={X: X_train, Y: Y_train}))
        return parameters

def pred():
    """
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    :return:
    """

if __name__=='__main__':
    train_X, test_X, train_Y, test_Y,name = loadDataSet('./DATG1/VTE4.txt','./DATG1/Y/gamma.txt')
    parameters=model(train_X.T,train_Y.T,test_X.T,test_Y.T)
    W=parameters['W']
    b=parameters['b']
    res = {}
    for i in range(W.shape[0]):
        res[i]=np.abs(W[i])
    res = sorted(res.items(), key=lambda d: d[1], reverse=True)
    rank = 1
    for key in res:
        print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank+=1