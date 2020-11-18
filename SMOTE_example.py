#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:28:37 2019

@author: xubing
"""

import pandas as pd
from collections import Counter
file = 'DATA_0724_com.csv'

df = pd.read_csv(file)

label = df['FS']
feature = df.drop('FS',axis=1)
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(label))


# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE

# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)

new_feature, new_label = smo.fit_sample(feature, label)
print(Counter(new_label))


#合并新数据
new_feature_df = pd.DataFrame(new_feature,columns=feature.columns)
new_label_df = pd.DataFrame(new_label,columns=[label.name])
new_df = pd.concat([new_feature_df,new_label_df],axis=1)
new_df = new_df.sample(frac=1).reset_index(drop=True)#打乱顺序
new_df.to_csv('DATA_0724_balance.csv',index=False,encoding='utf8')
