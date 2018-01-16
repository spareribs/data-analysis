# -*- coding: utf-8 -*-
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # logging是用来输出训练日志
# 分好词的句子，每个句子以词列表的形式输入
sentences = [['first', 'sentence'], ['second', 'sentence']]
# 用以上句子训练词向量模型
model = gensim.models.Word2Vec(sentences, min_count=1)
print(model['sentence'])  # 输出单词sentence的词向量。
'''
[  1.95254164e-04   4.48961137e-03   2.47035641e-03  -1.67108979e-03
   2.90901959e-03  -4.76417039e-03   1.36234227e-03  -3.45149217e-03
   4.06004721e-03   1.84493227e-04  -3.22105479e-03  -2.65526120e-03
  -1.40578067e-03  -1.22652898e-04   1.36450317e-03  -3.06492997e-03
   9.13523661e-04  -2.59605749e-03  -3.58700869e-03   4.89015458e-03
  -2.18498497e-03  -1.51738699e-04  -3.26284533e-03   3.27947317e-03
   4.79274290e-03   1.38248876e-03   1.24918739e-03  -2.98371934e-03
   4.79946611e-03  -1.07406208e-03  -4.65757539e-03  -3.84632312e-03
   4.81858943e-03  -2.10037455e-03   1.22576673e-03  -1.72215467e-03
  -1.18543452e-03   3.80422734e-03  -2.22733477e-03   3.16013442e-03
  -2.39789602e-03   9.74934781e-04   3.12719238e-03  -8.08136887e-04
   2.10699392e-03   3.08840745e-03  -2.65696738e-03   4.10316046e-03
   3.72426561e-03  -4.79483465e-03  -1.31346821e-03   3.17660946e-04
   4.63397475e-03  -4.93124826e-03   7.09711385e-05   3.60962190e-03
   4.28801263e-03   4.43177670e-03  -9.03548207e-04   4.51563206e-03
   2.42856937e-03   1.52298983e-03  -4.86341771e-03  -4.81435983e-03
  -4.41750605e-03  -3.26217414e-04  -3.12921102e-03  -2.81804893e-03
  -1.09485572e-03   1.48295972e-03   4.82936762e-03  -3.52945412e-03
   4.56330227e-03   3.29964329e-03  -4.12282161e-03  -1.46515679e-03
  -4.37626010e-03   8.87460832e-04  -6.95959956e-04   2.57947412e-03
  -4.45899117e-04  -3.31120403e-03  -4.01761197e-03   2.34769424e-03
  -2.90532061e-03  -1.23715191e-03  -3.84384766e-03  -8.53616628e-04
  -4.67098365e-03  -7.24320300e-04   2.74129701e-03   5.57981955e-04
   6.77675358e-04  -4.38690139e-03  -3.59827070e-03   4.22908831e-03
   1.35543523e-03   7.62621756e-04   9.29679314e-04   4.90467623e-03]
'''