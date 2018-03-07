#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/7 12:40
# @Author  : Spareribs
# @File    : decisiontree.py
"""

import operator
from math import log
from collections import Counter


def createDataSet():
    """DateSet 基础数据集
    Args: 无传入
    Returns: 数据集，标签
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
    :param dataSet: 数据集
    :param methodSelect: 选择处理的方法
    :return: 每一组feature下的某个分类下，香农熵的信息期望
    """

    ##########
    # 方法一：#
    ##########
    if methodSelect == "one":
        # 计算dataSet数据集的量
        numEntries = len(dataSet)
        print(u"[Info]：dataSet的类型为:{0},dataSet的长度为:{1}".format(type(dataSet), numEntries))

        # 计算分类标签label出现的次数
        labelCounts = {}
        for featVec in dataSet:
            # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
            currentLabel = featVec[-1]
            # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        print(u"[Info]：标签统计结果：{0}".format(labelCounts))

        # 对于label标签的占比，求出label标签的香农熵
        shannonEnt = 0.0
        for key in labelCounts:
            # 使用所有类标签的发生频率计算类别出现的概率。
            prob = float(labelCounts[key]) / numEntries
            # 计算香农熵，以 2 为底求对数
            shannonEnt -= prob * log(prob, 2)
        print(u"[Info]：香农熵：{0}".format(shannonEnt))

    ##########
    # 方法二：#
    ##########
    else:
        # 统计标签出现的次数
        label_count = Counter(data[-1] for data in dataSet)
        print(u"[Info]：结果统计字典：{0}".format(label_count))

        # 计算概率
        probs = [p[1] / len(dataSet) for p in label_count.items()]
        print(u"[Info]：{0}".format(probs))

        # 计算香农熵
        shannonEnt = sum([-p * log(p, 2) for p in probs])
        print(u"[Info]：{0}".format(shannonEnt))

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
        for featVec in dataSet:
            # index列为value的数据集【该数据集需要排除index列】
            # 判断index列的值是否为value
            if featVec[index] == value:
                # chop out index used for splitting
                # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
                reducedFeatVec = featVec[:index]
                reducedFeatVec.extend(featVec[index + 1:])
                # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
                # 收集结果值 index列为value的行【该行需要排除index列】
                retDataSet.append(reducedFeatVec)
        print(u"[Info]：数据集为：{0}".format(retDataSet))

    ##########
    # 方法二：#
    ##########
    else:
        # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
        pass
    return retDataSet


def chooseBestFeatureToSplit(dataSet, methodSelect="one"):
    """
    选择最好的特征
    :param dataSet: 数据集
    :param methodSelect: 选择处理的方法
    :return: 最优的特征列
    """

    ##########
    # 方法一：#
    ##########
    if methodSelect == "one":
        # 求第一行有多少列的 Feature, 最后一列是label列嘛
        numFeatures = len(dataSet[0]) - 1
        # label的信息熵
        baseEntropy = calcShannonEnt(dataSet)
        # 最优的信息增益值, 和最优的Featurn编号
        bestInfoGain, bestFeature = 0.0, -1
        # iterate over all the features
        for i in range(numFeatures):
            # create a list of all the examples of this feature
            # 获取每一个实例的第i+1个feature，组成list集合
            featList = [example[i] for example in dataSet]
            # get a set of unique values
            # 获取剔重后的集合，使用set对list数据进行去重
            uniqueVals = set(featList)
            # 创建一个临时的信息熵
            newEntropy = 0.0
            # 遍历某一列的value集合，计算该列的信息熵
            # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
            # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
            infoGain = baseEntropy - newEntropy
            print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature
    else:
        ##########
        # 方法二：#
        ##########
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


def majorityCnt(classList):
    """majorityCnt(选择出现次数最多的一个结果)

    Args:
        classList label列的集合
    Returns:
        bestFeature 最优的特征列
    """
    # -----------majorityCnt的第一种方式 start------------------------------------
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # print 'sortedClassCount:', sortedClassCount
    return sortedClassCount[0][0]
    # -----------majorityCnt的第一种方式 end------------------------------------

    # # -----------majorityCnt的第二种方式 start------------------------------------
    # major_label = Counter(classList).most_common(1)[0]
    # return major_label
    # # -----------majorityCnt的第二种方式 end------------------------------------


def createTree(dataSet, labels):
    """

    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print 'myTree', value, myTree
    return myTree


def classify(inputTree, featLabels, testVec):
    """

    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    """classify(给输入的节点，进行分类)

    Args:
        inputTree  决策树模型
        featLabels Feature标签对应的名称
        testVec    测试输入的数据
    Returns:
        classLabel 分类的结果值，需要映射label才能知道名称
    """
    # 获取tree的根节点对于的key值
    firstStr = inputTree.keys()[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


if __name__ == "__main__":
    dataSet, labels = createDataSet()
    shannonEnt = calcShannonEnt(dataSet)
    splitDataSet(dataSet, 2, "yes")
    print("*"*100)
    print(chooseBestFeatureToSplit(dataSet))
