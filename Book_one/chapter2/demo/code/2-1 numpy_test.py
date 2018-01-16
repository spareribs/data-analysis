# -*- coding: utf-8 -*-
import numpy as np  # 一般以np作为numpy的别名

a = np.array([2, 0, 1, 5])  # 创建数组
print(a, type(a))  # 输出数组[2 0 1 5] <class 'numpy.ndarray'>
print(a[:3], type(a[:3]))  # 引用前三个数组（切片）[2 0 1] <class 'numpy.ndarray'>
print(a.min(), type(a.min()))  # 输出a的最小元素0 <class 'numpy.int32'>
a.sort()
print(a, type(a))  # 将a的元素从小到大排序，此操作直接修改a，因此这时候a为[0 1 2 5] <class 'numpy.ndarray'>
b = np.array([[1, 2, 3], [4, 5, 6]])  # 创建二位数组
print(b * b, type(b * b))  # 输出数组的平方阵[[ 1  4  9],[16 25 36]] <class 'numpy.ndarray'>
