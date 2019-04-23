import numpy as np
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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
    print(X.shape)
    print(Y.shape)
    #划分训练集,验证集和测试集
    indices=np.arange(m)
    #random_state表示每次生成的训练集和测试集都是固定的，也就是结果可以重复
    X_train, X_test, Y_train, Y_test,index1,index2  = train_test_split(X, Y,indices,test_size=0.2,random_state=42)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test,test_size=0.5,random_state=42)
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape,X_valid.shape,Y_valid.shape)

    return X_train, X_test, X_valid,Y_train, Y_test,Y_valid, name , index1

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
def initialize_parameters(n):
    tf.set_random_seed(1)
    Z0 = tf.get_variable("Z0", shape=[1,2*n], initializer=tf.contrib.layers.xavier_initializer(seed=1))  #初始化X_feature和X_knockoffs的的系数
    Zb0 = tf.get_variable("Zb0", shape=[1, 2*n], initializer=tf.zeros_initializer())
    W0 = tf.get_variable("W0", shape=[1,n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b0= tf.get_variable("b0", shape=[1, n], initializer=tf.zeros_initializer())
    W1 = tf.get_variable("W1", shape=[n, n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", shape=[n, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", shape=[n,n ], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", shape=[n, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", shape=[1,n], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", shape=[1, 1], initializer=tf.zeros_initializer())


    parameters = {"Z0":Z0,
                  "Zb0":Zb0,
                  "W0":W0,
                  "b0":b0,
                  "W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
    }
    return parameters


#前向传播
def forward_propagation(X,parameters,n,lambd):
    print(type(n), n)
    Z0=  parameters['Z0']
    Zb0= parameters['Zb0']
    W0=  parameters['W0']
    b0=  parameters['b0']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    #正则化
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(W1))
    #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(W2))
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(W3))

    X_temp = tf.multiply(Z0, tf.transpose(X))
    X0=X_temp[:,0:n]
    X1=X_temp[:,n:2*n]
    X0=tf.add(X0,X1)

    # x0*w0+x0_knockoffs*w0'+x1*w1+x1_knockofss*w1'....
    A0=tf.add(tf.multiply(W0,X0),b0)
    A0=tf.transpose(A0)

    Z1 = tf.add(tf.matmul(W1, A0), b1)  # Z1 = np.dot(W1, X) + b1

    #隐藏层数目的选择
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    #Z2 = tf.nn.dropout(Z2, keep_prob=0.65)


    #使用sigmoid或者relu
    A2=tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, Z2 ), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def compute_cost(Z3, Y):

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)


    #如果是二分类
    #cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    """
    #如果是多分类任务
    #cost = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(Z3, 1e-10, 1.0)))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    """
    #如果是回归问题
    cost = tf.reduce_mean(tf.square(labels - logits))

    tf.add_to_collection('losses', cost)
    cost=tf.add_n(tf.get_collection('losses'))

    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: 打乱顺序
    permutation = list(np.random.permutation(m))#会返回一个长度为m的随机数组，且里面的数是0-m-1
    shuffled_X = X[:, permutation]  #将每列的数据按permutation的顺序来重新排列
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: 分割
    # num_complete_minibatches = math.floor(m/mini_batch_size) # original　
    num_complete_minibatches = int(math.floor(m / mini_batch_size))  # 一共有多少个集合
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # 如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
    # 如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，把剩下的放在一个集合内
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(X_train, Y_train , X_test, Y_test,X_valid,Y_valid, learning_rate=0.0001 ,
          minibatch_size=10, num_epochs=2000, print_cost=True):
    """
       输入参数:
       X_train -- 训练集, (输入特征数 ， 样例数 )
       Y_train -- 训练集, (输出维度 , 样例数 )
       X_test -- 测试集, (输入特征数  , 样例数 )
       Y_test -- 测试集, (输入特征数  , 样例数 )
       learning_rate -- 参数更新的learning rate
       minibatch_size -- 每个集合的样本数目,num_minibatch表示一共有多少个集合。我们在每个集合中选择一个进行训练
       num_epochs -- 迭代次数
       print_cost -- 是否每迭代100输出代价

    """
    tf.set_random_seed(1)
    seed = 3
    (n_x,m)=X_train.shape
    n_y=Y_train.shape[0]
    costs=[]
    #创建Placeholders,一个张量
    X,Y=create_placeholders(n_x,n_y)
    """下面的计算只是定义了一种计算的形式，并没有具体的数字，在实际run中使用这些函数来进行计算"""
    #初始化参数
    parameters=initialize_parameters(int(n_x/2))
    #前向传播
    Z3=forward_propagation(X,parameters,int(n_x/2),0.01)
    #计算代价
    cost=compute_cost(Z3,Y)

    # 后向传播: 定义tensorflow optimizer对象，这里使用AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #初始化所有参数
    init=tf.global_variables_initializer()

    #启动session来计算tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            """
             #对于样本数目很多的情况，可以使用mini_batch
            epoch_cost=0  #定义每一次迭代的代价
            num_minibatches=int(m/minibatch_size)  #训练集minibatch的数量
            seed=seed+1
            minibatches=random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y)=minibatch
                _,minibatch_cost=sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            """



            #进行批度训练
            epoch_cost=sess.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train})
            valid_cost = sess.run(cost, feed_dict={X: X_valid, Y: Y_valid})
            #print(epoch_cost)
            epoch_cost=epoch_cost[1]

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                print("valid_cost: ",valid_cost)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        #神经网络经过训练后得到的值
        Z3 = sess.run(Z3,feed_dict={X: X_train,Y:Y_train })
        print('train cost: ',sess.run(cost,feed_dict={X: X_train,Y:Y_train }))
        #模型训练完成后，测试在测试集上的表现
        cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        print('test cost: ', cost)


        return parameters

def analyse(parameters,n):
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
    #W = np.matmul(W3, W2)
    W=np.matmul(W3,W1)
    #W=W3

    Z1 = np.multiply(Z0[:, 0:n], W0)
    Z2 = np.multiply(Z0[:, n:2*n], W0)

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

    rank = 1
    for key in res:
        #print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if(name[key[0]] == '90'or name[key[0]]=='91'or name[key[0]]=='92' or name[key[0]]=='93' or name[key[0]]=='94' or name[key[0]]=='95'
                or name[key[0]]=='96' or name[key[0]]=='97' or name[key[0]]=='98' or name[key[0]]=='99'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or
                name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1



    rank = 1
    for key in res1:
        #print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == '90' or name[key[0]] == '91' or name[key[0]] == '92' or name[key[0]] == '93' or name[
            key[0]] == '94' or name[key[0]] == '95'
                or name[key[0]] == '96' or name[key[0]] == '97' or name[key[0]] == '98' or name[key[0]] == '99'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or
                name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1


    fr=open('./result.txt','a')
    res=[]
    rank = 1
    for key in res2:
        #print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == '90' or name[key[0]] == '91' or name[key[0]] == '92' or name[key[0]] == '93' or name[
            key[0]] == '94' or name[key[0]] == '95'
                or name[key[0]] == '96' or name[key[0]] == '97' or name[key[0]] == '98' or name[key[0]] == '99'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        #fr.write(str(rank)+'    '+str(name[key[0]])+'   '+str(key[1])+'\n' )
        if (name[key[0]] == 'GRMZM5G872256' or name[key[0]] == 'GRMZM2G066734' or name[key[0]] == 'GRMZM2G012455' or
                name[key[0]] == 'GRMZM2G138589' or name[key[0]] == 'GRMZM2G004528' or name[key[0]] == 'GRMZM5G870176' or
                name[key[0]] == 'GRMZM2G015040'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        if (name[key[0]] == 'GRMZM2G324886' or name[key[0]] == 'GRMZM2G150906' or name[key[0]] == 'GRMZM2G158232' or
                name[key[0]] == 'GRMZM2G082780' or name[key[0]] == 'GRMZM2G039454'):
            print(rank, '(', key[0], ' ,', name[key[0]], ' ,', key[1])
        rank += 1

def predict(X_test,Y_test,parameters):
    Z0 = parameters['Z0']
    W0 = parameters['W0']
    b0 = parameters['b0']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # 计算下测试集的误差

    m,n=X_test.shape
    X_k = np.zeros((m, n))
    X_test = np.c_[X_test, X_k]
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    Y_test=np.transpose(Y_test)
    print(X_test.shape, Y_test.shape)
    X_temp = tf.multiply(Z0, X_test)
    X0 = X_temp[:, 0:n]
    X1 = X_temp[:, n:2 * n]
    X0 = tf.add(X0, X1)
    A0 = tf.add(tf.multiply(W0, X0), b0)
    A0 = tf.transpose(A0)
    Z1 = tf.add(tf.matmul(W1, A0), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    # Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    ##A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A1), b3)  # Z3 = np.dot(W3,Z2) + b3
    cost = tf.reduce_mean(tf.square((Y_test) - Z3))
    sess = tf.Session()
    print("Z0: ",Z0);print("W0: ",W0);print("b0: ",b0);print("W1: ",W1);print("b1: ",b1);print("W2: ",W2);print("b2: ",b2);print("W3: ",W3);print("b3: ",b3)
    print("Z3: ",sess.run(Z3))
    print("Y :",Y_test)
    print('predict_cost: ', sess.run(cost))

def normlization(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X= (X - mu) / (std+0.001)
    return X

#扩展测试集和验证集
def expand(X,Y):
    m, n = X.shape
    X_k = np.zeros((m, n))
    X = np.c_[X, X_k]
    X = np.transpose(X)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return X,Y

if __name__=="__main__":
    #得到训练集和测试集
    X_train_orig, X_test_orig, X_valid_orig, Y_train_orig, Y_test_orig,Y_valid_orig,name,index1=loadDataSet('./breast_cancer/X1.txt','./breast_cancer/Y.txt')

    #生成X对应的knockoffX数据
    knockoffX=[]
    fr =open("./breast_cancer/X1-knockoffs.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #print(curLine)
        fltLine = list(map(float, curLine))
        knockoffX.append(fltLine)
    knockoffX=np.mat(knockoffX)
    #只取训练集那一部分
    knockoffX=knockoffX[index1,:]

    print('knockoffX.shape: ',knockoffX.shape)

    X_train_orig = np.hstack((X_train_orig, knockoffX))


    # 归一化
    #X_train_orig = (X_train_orig + 3) / 6  #
    # 标准化
    #X_train_orig =  normlization(X_train_orig)
    #X_test_orig =  normlization(X_test_orig)
    #X_valid_orig =  normlization(X_valid_orig)
    #Y_train_orig =  normlization(Y_train_orig)
    #Y_test_orig =  normlization(Y_test_orig)
    #Y_valid_orig =  normlization(Y_valid_orig)

    X_train_orig=normalize(X_train_orig,norm='l2')
    X_test_orig=normalize(X_test_orig,norm='l2')
    X_valid_orig=normalize(X_valid_orig,norm='l2')
    Y_train_orig=normalize(Y_train_orig,norm='l2')
    Y_test_orig=normalize(Y_test_orig,norm='l2')
    Y_valid_orig=normalize(Y_valid_orig,norm='l2')



    print('X_train_orig.shape: ',X_train_orig.shape)
    m,n=X_train_orig.shape
    # 把训练集和测试集的图像展开
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T

    X_train = X_train_flatten
    X_test = X_test_orig
    X_valid =X_valid_orig

    Y_train=Y_train_orig
    Y_test=Y_test_orig
    Y_valid=Y_valid_orig

    X_test,Y_test=expand(X_test,Y_test)
    X_valid,Y_valid=expand(X_valid,Y_valid)


    print(X_train.shape,X_test.shape,X_valid.shape)
    parameters=model(X_train,Y_train.T,X_test,Y_test.T,X_valid,Y_valid.T)

    #统计结果
    analyse(parameters,int(n/2))












