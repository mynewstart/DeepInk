import csv
import numpy as np


def loaddata(file):
    fr = open(file, 'r', encoding='utf-8-sig')
    reader = csv.reader(fr)
    data = []
    fltLine = []
    for line in reader:
        data.append(line)
    data = np.mat(data)
    data = data.astype(np.float32)
    day = data[:, 0]
    y = data[:, 1]
    index = np.where(data == 1)
    y = y[index]
    return index[0], y


def load(file, index):
    fr = open(file, 'r', encoding='utf-8-sig')
    reader = csv.reader(fr)
    data = []
    fltLine = []
    rank = 0
    for line in reader:
        rank += 1
        data.append(line)
    data = np.mat(data)
    data = data.astype(np.float32)
    return data[:, index]


if __name__ == '__main__':
    index, y = loaddata("C:/Users/DELL/Desktop/Day.csv")
    print(index)
    print(type(y))
    np.savetxt('C:/Users/DELL/Desktop/Y.csv', y.T)
    # data=load("C:/Users/DELL/Desktop/1.csv",index)
