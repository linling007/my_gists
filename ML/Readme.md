# Author by xubing on 2020-06
# 使用说明
```
泰坦尼克数据的使用样例

import numpy as np
import pandas as pd

from general_models_training import GeneralModelsTraining

# Step1: 读数据
file = 'assets/titanic.csv'
df = pd.read_csv(file)
target = 'Survived'

# Step2: 数据处理（入模数据准备）
x = df.drop(target, axis=1)
y = df[target]
x = x.replace({np.inf: 0, np.nan: 0, -np.inf: 0})
x = x.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Step3: 模型训练与结果查看
gmt = GeneralModelsTraining(x, y, model_name='dt') ## 选择算法,可选算法为 'lr'、'dt'、'rf'、'gbdt'、'xgboost'、# 不平衡样本可选参数为 'bbc'、'brc'

```
**Tips:**

> 超参数搜索范围在config.py 文件中修改。在此文件中可自定义超参数搜索。
> 在文末重写可覆盖原始超参数。

