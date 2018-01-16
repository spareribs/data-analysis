# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression  # 导入线性回归模型

model = LinearRegression()  # 建立线性回归模型
print(model, type(model))
'''
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
<class 'sklearn.linear_model.base.LinearRegression'>
'''

from sklearn import datasets  # 导入数据集

iris = datasets.load_iris()  # 加载数据集
print(iris.data.shape, type(iris.data.shape))  # 查看数据集大小(150, 4) <class 'tuple'>
from sklearn import svm  # 导入SVM模型

clf = svm.LinearSVC()  # 建立线性SVM分类器
clf.fit(iris.data, iris.target)  # 用数据训练模型
clf.predict([[5.0, 3.6, 1.3, 0.25]])  # 训练好模型之后，输入新的数据进行预测
clf.coef_  # 查看训练好模型的参数
