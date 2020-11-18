# -*- coding: utf-8 -*-
"""
Created by xubing on 2020/4/26

"""
scoring = 'recall'  # recall precision f1
cv = 4  # 交叉验证折数
test_size = 0.25  # 测试集比例
random_state = 43  # 分割数据的随机种子
imblearn_model = 'brc'  # 非平衡样本时选择的算法
normal_model = 'lr'  # 平衡样本时选择的算法



# grid search cross validation hyper params
lr_gs_cv_hyper_params = {
    'penalty': ['l2'],  # 'l1',
    'C': [0.001, 0.1, 1, 1.5, 2],
    'solver': ['sag', 'saga']

}
dt_gs_cv_hyper_params = {
    'criterion': ['gini', 'entropy'],
    # 'splitter': ['random', 'best'],
    # 'max_depth': [2, 3, 5, 7, 8, 9, 20],
    'max_depth': [3, 5, 7, 9, 10, 12, 20],
    'min_samples_split': [0.1,  0.3, 0.5],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_weight_fraction_leaf': [0, 0.01, 0.1, 0.5],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced'],
    # 'max_leaf_nodes':
    # 'min_impurity_decrease'

}
dt_gs_cv_hyper_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 3, 4, 5, 7, 10, 15],
    'min_samples_split': [2, 4, 6, 8],
    'min_weight_fraction_leaf': [0, 0.01, 0.1, 0.5],
    'max_leaf_nodes': [2, 5, 10],
    'class_weight': ['balanced', None]
}

rf_gs_cv_hyper_params = {
    'criterion': ['gini'],
    'max_depth': [20, 30],
    'min_samples_split': [2],
    'n_estimators': [100, 150, 200],
    'min_weight_fraction_leaf': [0, 0.1],
    'max_leaf_nodes': [20],
    'class_weight': ['balanced']
}
gbdt_gs_cv_hyper_params = {
    'max_depth': [20, 30],
    'min_samples_split': [2],
    'n_estimators': [100, 150, 200],
    'min_weight_fraction_leaf': [0.1, 0.5],
    'max_leaf_nodes': [30, None],
    'loss': ['deviance'],
    'learning_rate': [0.3],
    'max_features': ['sqrt']
}
xgboost_gs_cv_hyper_params = {

}
brc_gs_cv_hyper_params = {
    'max_depth': [1, 2, 3, 5, 7, 32, None],
    'n_estimators': [100, 200, 300, 500],
    # 'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3],
    # 'class_weight': ['balanced', 'auto']

}
bbc_gs_cv_hyper_params = {}
rusbc_gs_cv_hyper_params = {}
eec_gs_cv_hyper_params = {}
