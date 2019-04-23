import numpy as np
import csv
import xlwt


#将矩阵写入到excel中
def save(data,path):
    f=xlwt.Workbook()#创建工作簿
    sheet1=f.add_sheet(u'sheet1',cell_overwrite_ok=True)
    h,l=data.shape
    for i in range(h):
        for j in range(l):
            sheet1.write(i,j,data[i,j])

    f.save(path)

def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)  #升序
    sortArray=sortArray[-1::-1]  #降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def pca(X,percentage=0.99):
    meanVal=np.mean(X,axis=0) #按列求取特征值
    newX=X-meanVal
    covMat=np.cov(newX,rowvar=0) #求协方差矩阵，0表示一行一个样本
    print(covMat.shape)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) #求特征值和特征向量
    #print(len(eigVals),eigVals)
    #print(eigVects)
    n=percentage2n(eigVals,percentage)
    print(n)
    eigValIndice=np.argsort(eigVals) #特征值从小到大排序
    rank=0
    for i in range(len(eigVals)):
        if eigVals[i]<1e-2:
            rank+=1
            #print(i)
    print(rank)
    print(eigValIndice)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1] #最大n个特征值的下标
    print(n_eigValIndice)
    n_eigVect=eigVects[:,n_eigValIndice]  #对应的特征向量
    lowDDataMat=newX*n_eigVect  #m*n和n*k的矩阵相乘，得到m*k的矩阵
    np.savetxt('low.csv', lowDDataMat, delimiter = ',')
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal
    np.savetxt('recon.csv', reconMat, delimiter = ',')
    return lowDDataMat,reconMat
def loadData():
    file=open("F:/DeepInk/GY-DATA/500.txt","r")
    X=[]
    for line in file.readlines():
        curLine =line.strip().split('\t')
        # 字符型转化为浮点型
        fltLine = list(map(float, curLine))
        X.append(fltLine)
    return np.mat(X)

if __name__=='__main__':
    #X=loadData()

    X=[[4,2,-5],[6,4,-9],[5,3,-7]]
    X=np.mat(X)

    print(np.linalg.eig(X))


    #U,S,V=np.linalg.svd(X)
    #print(U)
    #print(S)
    #print(V)
    #lowData,recData=pca(X)
    #print(lowData)
    #print(recData)