'''
create by xubing

'''

# import pandas as pd
#
# from general_ml_model import GeneralMLModel
#
# df = pd.read_csv('bank_train.csv')
# target = 'y'
# df = df.drop([col for col in df.columns if df[col].dtype == 'object'], axis=1)
# X = df.drop(target, axis=1)
# y = df[target]
#
# gm = GeneralMLModel(X, y)
# gm.auto_tune_gbdt()


from pprint import pprint as pp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from general_models_training import GeneralModelsTraining
from utils import PlottingTool

raw_data = pd.read_excel("./raw_data_20171231_0425-特征分析.xlsx")

raw_data_drop_oneline = raw_data[~(raw_data['me00000001'] == '是否央企')]

# 去零
raw_data_drop_oneline.replace(np.nan, 0, inplace=True)
raw_data_drop_oneline.replace([np.inf, -np.inf], [0, 0], inplace=True)

raw_data_drop_oneline_columns = raw_data_drop_oneline.drop(['Unnamed: 0', 'entid', 'entn75me', 'ish75vebd', 'time'],
                                                           axis=1)

market_risk = pd.DataFrame(raw_data_drop_oneline_columns,
                           columns=['islistsk', 'me00000001', 'me10000001', 'me10000002', 'me10000003', 'me20200001',
                                    'me20200002', 'me20200005', 'me20300001', 'me20300002', 'me20300003', 'me20300005',
                                    'me20300006', 'me20300004', 'me20300007', 'me20000002', 'me20000003', 'me20200006',
                                    'me20200007', 'me20300008', 'me20300009', 'me20300010', 'me20400001', 'me20500001',
                                    'me20500002', 'me20600001', 'isbddflt'])

# market_risk

# 财务风险
finance_risk = pd.DataFrame(raw_data_drop_oneline_columns,
                            columns=['e50000009', 'e50000010', 'e50000028', 'e50000003', 'e50000015', 'e50000020',
                                     'e50300004', 'e50000029', 'e50000021', 'e50000012', 'e50000030', 'e50000031',
                                     'e50000032', 'e50000033', 'e50000034', 'e50000035', 'e50000027', 'e50000036',
                                     'e50000038', 'e50000039', 'e50200001', 'e50200002', 'e50200012', 'e50200003',
                                     'e50200004', 'e50200018', 'e50200019', 'e50200020', 'e50100030', 'e50100019',
                                     'e50100021', 'e50100031', 'e50400011', 'e50400002', 'e50400005', 'e50400007',
                                     'e50400003', 'e50400012', 'e50400013', 'e50400015', 'e50800001', 'e50800002',
                                     'e50800003', 'e50200006', 'e50700024', 'e50700026', 'e50700027', 'e50700029',
                                     'e50700031', 'e50700033', 'e50700035', 'e50700036', 'e50700037', 'e50300013',
                                     'e50300014', 'e50300015', 'e50300016', 'e50300017', 'e50300018', 'e50300019',
                                     'e50300020', 'e50300021', 'e50500008', 'e50500009', 'e50500010', 'e50500011',
                                     'e50500012', 'e50500013', 'e50500014', 'e50500015', 'isbddflt'])

finance_risk

# 全部风险特征
market_finance_risk = raw_data_drop_oneline_columns

target = 'isbddflt'
X = market_finance_risk.drop(target, axis=1)
y = market_finance_risk[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
GMT = GeneralModelsTraining(x_train, x_test, y_train, y_test)
result = GMT.universal_model_training('dt')
pp(result)

pt = PlottingTool(result['tuned_model'], x_train, x_test, y_train, y_test)
pt.plot_roc()
pt.plot_pr()
pt.plot_confusion_matrix()
# pt.plot_dt_graph() 只有决策树才能绘制决策图
