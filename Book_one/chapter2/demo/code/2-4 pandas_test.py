# -*- coding: utf-8 -*-
import pandas as pd  # 通常用pd作为pandas的别名。

s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])  # 创建一个序列s
print(s, type(s))
'''
a    1
b    2
c    3
dtype: int64 <class 'pandas.core.series.Series'>
'''

d = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])  # 创建一个表
print(d, type(d))
'''
   a  b  c
0  1  2  3
1  4  5  6 <class 'pandas.core.frame.DataFrame'>
'''

d2 = pd.DataFrame(s)  # 也可以用已有的序列来创建表格
print(d2, type(d2))
'''
   0
a  1
b  2
c  3 <class 'pandas.core.frame.DataFrame'>
'''

d.head()  # 预览前5行数据
print(d.head(), type(d.head()))
'''
   a  b  c
0  1  2  3
1  4  5  6 <class 'pandas.core.frame.DataFrame'>
'''
d.describe()  # 数据基本统计量vv
print(d.describe(), type(d.describe()))
'''
             a        b        c
count  2.00000  2.00000  2.00000
mean   2.50000  3.50000  4.50000
std    2.12132  2.12132  2.12132
min    1.00000  2.00000  3.00000
25%    1.75000  2.75000  3.75000
50%    2.50000  3.50000  4.50000
75%    3.25000  4.25000  5.25000
max    4.00000  5.00000  6.00000 <class 'pandas.core.frame.DataFrame'>
'''

# 读取文件，注意文件的存储路径不能带有中文，否则读取可能出错。
pd_excel = pd.read_excel('../data/data.xlsx')  # 读取Excel文件，创建DataFrame。
print(pd_excel, type(pd_excel))
'''
Columns: [这是一个测试excel。]
Index: [] <class 'pandas.core.frame.DataFrame'>
'''

pd_csv = pd.read_csv('../data/data.csv', encoding='utf-8')  # 读取文本格式的数据，一般用encoding指定编码。
print(pd_csv, type(pd_csv))
'''
Columns: [test]
Index: [] <class 'pandas.core.frame.DataFrame'>
'''