# -*- coding: utf-8 -*-
from statsmodels.tsa.stattools import adfuller as ADF  # 导入ADF检验
import numpy as np

ADF(np.random.rand(100))  # 返回的结果有ADF值、p值等
print(ADF(np.random.rand(100)))
'''
(-9.4941503607598374, 3.5863098635670053e-16, 0, 99,
{   '5%': -2.8912082118604681,
    '1%': -3.4981980821890981,
    '10%': -2.5825959973472097},
43.974101581213972)

'''