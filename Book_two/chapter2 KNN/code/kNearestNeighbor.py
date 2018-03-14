#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/1/25 17:35
# @Author  : Spareribs
# @File    : kNearestNeighbor.py
"""
import os
import operator
from numpy import *


def getdatapath(filename=""):
    # print u"当前目录[含文件名] 绝对路径：{0}".format(os.path.abspath(__file__))
    # print u"当前目录[不含文件名] 绝对路径：{0}".format(os.path.dirname(os.path.abspath(__file__)))
    # print u"返回上一级目录：{0}".format(os.path.pardir)
    # print u"父级目录demo 绝对路径：{0}".format(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    demo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    data_path = os.path.join(demo_path, "data", filename)
    return data_path


def file2matrix(filename):
    """
    导入训练数据
    :param filename: 数据文件路径
    :return: 数据矩阵return_Mat 和 对应的类别 classLabel_Vector
    """
    fr = open(filename)
    number_Of_Lines = len(fr.readlines())  # 获得文件中的数据行的行数,用于生成数据矩阵
    return_Mat = zeros((number_Of_Lines, 3))  # 生成对应的空矩阵
    classLabel_Vector = []  # 存储各项行数据的标记标签
    fr = open(filename)
    index = 0  # return_Mat 和 classLabel_Vector 的偏移量标记
    for line in fr.readlines():
        list_From_Line = line.strip().split('\t')  # 处理后获取每行的数据
        return_Mat[index, :] = list_From_Line[0:3]  # 前3个值存入数据矩阵
        classLabel_Vector.append(int(list_From_Line[-1]))  # 最后1个值存入类别数据
        index += 1  # 偏移量标记加1
    return return_Mat, classLabel_Vector


def autoNorm(dataSet):
    """
    归一化特征值，消除属性之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """

    minVals = dataSet.min(0)  # 计算每种属性的最小值
    maxVals = dataSet.max(0)  # 计算每种属性的最大值
    ranges = maxVals - minVals  # 极差
    m = dataSet.shape[0]  # 获取矩阵dataSet一维的长度
    normDataSet = dataSet - tile(minVals, (m, 1))  # 生成与最小值之差组成的矩阵，tile用minVals生成二维长度为1000的矩阵
    normDataSet1 = normDataSet / tile(ranges, (m, 1))  # 将最小值之差除以范围组成矩阵
    return normDataSet1, ranges, minVals


def classify0(inX, dataSet, labels, k):
    """
    inx[1,2,3]
    DS=[[1,2,3],[1,2,0]]
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """

    # -----------实现 classify0() 方法的第一种方式----------------------------------------------------------------------------------------------------------------------------
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数

    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort()
    # print 'distances.argsort()=', sortedDistIndicies

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def main():
    # 步骤1：导入训练数据，获得数据矩阵return_Mat 和 对应的类别 classLabel_Vector
    filename = getdatapath("datingTestSet2.txt")
    return_Mat, classLabel_Vector = file2matrix(filename)
    # print return_Mat, classLabel_Vector

    # 步骤2：归一化特征值，消除属性之间量级不同导致的影响
    # 这个步骤的数据处理有待进一步学习
    normMat, ranges, minVals = autoNorm(return_Mat)

    # 步骤3：对约会网站的测试方法
    hoRatio = 0.05  # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    m = normMat.shape[0]  # m 表示数据的行数，即矩阵的第一维
    numTestVecs = int(m * hoRatio)  # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    print(u'用于测试的数据量: {0}'.format(numTestVecs))
    errorCount = 0  # 统计错误数量
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], classLabel_Vector[numTestVecs:m], 3)
        print(u"the classifier came back with: %d, the real answer is: %d" % (classifierResult, classLabel_Vector[i]))
        if classifierResult != classLabel_Vector[i]:
            errorCount += 1
    print(u'测试的数量:{0}\n错误的数量:{1}\n错误的比率:{2}'.format(numTestVecs, errorCount, errorCount / float(numTestVecs)))


if __name__ == '__main__':
    main()
