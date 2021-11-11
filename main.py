import numpy as np
import operator
import os

def knn(k,testdata, traindata, labels):
    traindatasize = traindata.shape[0]
    dif = np.tile(testdata, (traindatasize, 1)) - traindata
    #计算距离
    sqdif = dif**2
    sumsqdif = sqdif.sum(axis=1)
    distance = sumsqdif**0.5
    sortdistance = distance.argsort()
    count={}
    for i in range(k):
        vote = labels[sortdistance[i]]
        count[vote] = count.get(vote, 0)+1
    sortcount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortcount[0][0]

#加载数据，将文件转化为数组形式
def datatoarray(filename):
    arr = []
    fh = open(filename)
    for i in range(32):
        thisline = fh.readline()
        for j in range(32):
            arr.append(int(thisline[j]))
    return arr

#获取文件的label
def get_labels(filename):
    label = int(filename.split('_')[0])
    return label

#建立训练数据
def train_data():
    labels = []
    trainlist = os.listdir('traindata/')
    num = len(trainlist)
    trainarr = np.zeros((num, 1024))
    for i in range(num):
        thisfile = trainlist[i]
        labels.append(get_labels(thisfile))
        trainarr[i,:] = datatoarray("traindata/"+thisfile)
    return trainarr,labels

#用测试数据调用KNN算法进行测试
def datatest():
    a = [] #准确结果
    b = [] # 预测结果
    traindata, labels = train_data()
    testlist = os.listdir('testdata/')
    fh = open('result_knn.csv', 'a')
    for test in testlist:
        testfile = 'testdata/'+test
        testdata = datatoarray(testfile)
        result = knn(3, testdata,traindata,labels)
        #将预测结果存于文本中
        fh.write(test+'--------'+str(result)+'\n')
        a.append(int(test.split('_')[0]))
        b.append(int(result))
    fh.close()
    return a,b

if __name__ == "__main__":
    a,b = datatest()
    num = 0
    for i in range(len(a)):
        if(a[i]==b[i]):
            num+=1
        else:
            print("预测失误：", a[i], "预测为", b[i])
    print("预测样本数为：", len(a))
    print("预测成功数为：", num)
    print("模型准确率为：", num/len(a))


