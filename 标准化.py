import numpy as np
from scipy import stats
from scipy.stats import kstest
from scipy.stats import normaltest

def normalization(fr):
    fr = open(fr)
    X=[]
    lines=fr.readlines()
    name=lines[0]
    for i in range(1,len(lines)):
        curLine = lines[i].strip().split('\t')
        X.append(list(map(float,curLine)))
    X=np.mat(X)
    m,n=X.shape

    """
    #标准化
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X=(X-mu)/std
    np.savetxt("./代谢物/chr3-norm.txt",X)
    """

    return X

if __name__=='__main__':
   X=[[1,2,3],[1,2,3]]
   X=np.mat(X)
   print(X.shape)
   mu = np.mean(X, axis=1)
   print(mu)
   std = np.std(X, axis=1)
   print(std)
   X = (X - mu) / std
   print(X)