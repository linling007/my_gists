"""
Created by xubing on 2020/6/11

"""

import pandas as pd

from general_models_training import GeneralModelsTraining

# 读数据
file = 'assets/sample_all_features3.5.xlsx'
df = pd.read_excel(file)
target = 'isbad'

# 数据预处理
x = df.drop(target, axis=1)
y = df[target]
x = x.fillna(0)
x = x.drop(['name', 'badtime'], axis=1)
for col in x.columns:
    x[col] = x[col].fillna(x[col].mean())
x = x[x.columns[11:]]

# 建模与评估
gmt = GeneralModelsTraining(x, y, model_name='rusbc')

