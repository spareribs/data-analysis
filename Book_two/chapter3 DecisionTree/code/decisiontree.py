#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/7 12:40
# @Author  : Spareribs
# @File    : decisiontree.py
"""
import json
import operator
from math import log
from collections import Counter


def createDataSet():
    """
    生成基础数据集
    :return: 数据集，标签列表
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet, methodSelect="one"):
    """
    计算给定数据集的香农熵
    注意：默认dataSet中最后一列的值为结果值
    :param dataSet: 数据集
    :param methodSelect: 选择处理的方法
    :return: 每一组feature下的某个分类下，香农熵的信息期望
    """

    #########
    # 方法一 #
    #########
    if methodSelect == "one":
        numEntries = len(dataSet)  # 计算数据集的量
        labelCounts = {}  # 计算分类标签label出现的次数，默认最后一列
        for featVec in dataSet:  # 为所有可能分类创建字典
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1

        shannonEnt = 0.0  # 对于label标签的占比，求出label标签的香农熵
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries  # 计算所有标签的发生频率计算类别出现的概率
            shannonEnt -= prob * log(prob, 2)  # 使用概率计算香农熵（以2为底求对数）

    #########
    # 方法二 #
    #########
    else:
        labelCounts = Counter(data[-1] for data in dataSet)  # 统计标签出现的次数
        probs = [p[1] / len(dataSet) for p in labelCounts.items()]  # 计算概率
        shannonEnt = sum([-p * log(p, 2) for p in probs])  # 计算香农熵

    return shannonEnt


def splitDataSet(dataSet, index, value, methodSelect="one"):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param index: 划分数据集特征的索引
    :param value: 需要返回的特征的值
    :param methodSelect: 选择处理的方法
    :return: index列为value的数据集【该数据集需要排除index列】
    """

    ##########
    # 方法一：#
    ##########
    if methodSelect == "one":
        retDataSet = []
        for featVec in dataSet:  # 循环获取每行的数据
            if featVec[index] == value:  # 判断索引中的值与给定的目标值是否相等
                reducedFeatVec = featVec[:index]  # 取索引前的所有数据
                reducedFeatVec.extend(featVec[index + 1:])  # 取索引后的所有数据
                retDataSet.append(reducedFeatVec)  # 将排除了索引值的结果存入新的retDataSet中

    ##########
    # 方法二：#
    ##########
    else:
        # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
        retDataSet = ""
        pass
    return retDataSet


def chooseBestFeatureToSplit(dataSet, methodSelect="one"):
    """
    选择最好的特征
    :param dataSet: 数据集
    :param methodSelect: 选择处理的方法
    :return: 最优的特征列
    """

    #########
    # 方法一 #
    #########
    if methodSelect == "one":
        numFeatures = len(dataSet[0]) - 1  # 求dataSet出特征的数量
        baseEntropy = calcShannonEnt(dataSet)  # 计算dataSet的信息熵（香农熵）
        bestInfoGain, bestFeature = 0.0, -1  # 设置最优的信息增益值, 和最优的特征编号
        for i in range(numFeatures):  # 通过循环迭代所有的特征
            featList = [example[i] for example in dataSet]  # 获取当前列的所有特征值，组成一个List
            uniqueVals = set(featList)  # 特征值去重
            newEntropy = 0.0  # 创建一个临时的信息熵
            for value in uniqueVals:  # 遍历去重后的特征
                subDataSet = splitDataSet(dataSet, i, value)  # 重新划分数据集
                prob = len(subDataSet) / float(len(dataSet))  # 计算新数据集占原数据集的比例
                newEntropy += prob * calcShannonEnt(subDataSet)  # 将所有的数据集计算的香农熵求和，得到信息熵
            infoGain = baseEntropy - newEntropy  # 差值越大，说明newEntropy越小，没那么混乱
            if infoGain > bestInfoGain:  # 比较所有特征中的信息增益，返回最好特征划分的索引值
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature
    else:
        #########
        # 方法二 #
        #########
        # 计算初始香农熵
        base_entropy = calcShannonEnt(dataSet)
        best_info_gain = 0
        best_feature = -1
        # 遍历每一个特征
        for i in range(len(dataSet[0]) - 1):
            # 对当前特征进行统计
            feature_count = Counter([data[i] for data in dataSet])
            # 计算分割后的香农熵
            new_entropy = sum(
                feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) for feature in
                feature_count.items())
            # 更新值
            info_gain = base_entropy - new_entropy
            print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature


def majorityCnt(classList, methodSelect="one"):
    """

    :param classList: 分类标签的列表
    :return: 返回出现次数醉倒的分类名称
    """

    #########
    # 方法一 #
    #########
    if methodSelect == "one":
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        # print 'sortedClassCount:', sortedClassCount
        return sortedClassCount[0][0]
    #########
    # 方法二 #
    #########
    else:
        major_label = Counter(classList).most_common(1)[0]
        return major_label


def createTree(dataSet, labels):
    """
    创建树的函数代码
    :param dataSet: 数据集
    :param labels: 标签列表
    :return:
    """
    classList = [example[-1] for example in dataSet]  # 数据集的所有类标签
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 存储最好特征的变量
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 当特征被加入到myTree中以后，删除该特征
    featValues = [example[bestFeat] for example in dataSet]  # 得到列表包含的所有属性值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    print(u"[Info]：当前创建出来的数据集为{0}, 结果标签为{1}".format(dataSet, labels))
    ################################
    # 这部分是用于测试各部分函数的例子 #
    ################################
    # # calcShannonEnt 函数测试
    # shannonEnt = calcShannonEnt(dataSet)
    # print(u"[Info]：数据集 {0} 的香农熵为 {1} .".format(dataSet, shannonEnt))
    # # splitDataSet 函数测试
    # retDataSet = splitDataSet(dataSet, 1, 1)
    # print(u"[Info]: 数据集 {0} 按照索引1的值为1的方式划分，得到新的数据集 {1}".format(dataSet, retDataSet))
    # # chooseBestFeatureToSplit 函数测试
    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print(u"[Info]: 通过最好的特征选择，索引为 {0} 的特征最好".format(bestFeature))

    myTree = createTree(dataSet, labels)
    print(u"[Info]：决策树为：\n{2}".format(dataSet, labels, json.dumps(myTree, indent=4, separators=(',', ': '))))
