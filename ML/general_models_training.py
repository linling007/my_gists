# -*- coding: utf-8 -*-
"""
Created by xubing on 2020/4/26

"""
import sys
import warnings
from pprint import pprint as pp

import numpy as np
from imblearn.ensemble import BalancedBaggingClassifier as bbc
from imblearn.ensemble import BalancedRandomForestClassifier as brc
from imblearn.ensemble import RUSBoostClassifier as rusbc
from imblearn.ensemble import EasyEnsembleClassifier as eec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config
from utils import PlottingTool
from utils import fn_timer

warnings.filterwarnings('ignore')


class GeneralModelsTraining:
    def __init__(self, x, y, model_name=None):
        self.func_name = 'GMT(GeneralModelsTraining)'
        self.x, self.y = x, y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=config.test_size,
                                                                                random_state=config.random_state)
        in_model = self.check_data()  # 检查一下数据是否符合入模标准
        if in_model:  # 如果不符合入模标准，会有不符合的提示。然后退出程序
            print('** 请根据提示重新处理数据!')
            sys.exit(0)
        else:
            print('** 数据检查通过，开始建模！')

        # 判断是否是二分类，如果不是二分类，添加一个多分类的类别数
        self.classes = len(y.value_counts().index.tolist())  # 类别数量
        print('* 分类任务为{0}分类'.format(self.classes))
        classes = self.classes
        if classes == 2:  # 如果是二分类，判断是否是正负样本是否平衡：差10倍为不平衡
            vc = y.value_counts().values.tolist()
            if vc[0] > vc[1]:
                large_class, small_class = vc[0], vc[1]
            else:
                large_class, small_class = vc[1], vc[0]
            if (large_class / small_class) > 10:
                model_name = 'brc' if model_name is None else model_name
                print('** 非平衡数据，进入非平衡数据处理模式，可选参数为"bbc"、"brc"')
                self.imblearn_classification(model_name=model_name)
            else:
                model_name = 'lr' if model_name is None else model_name
                print('** 进入平衡数据处理模式，可选参数为"lr"、"dt"...')
                self.universal_model_training(model_name=model_name)
        else:
            print("** 目前还不支持多分类，新功能很快上线")
            sys.exit(0)

    def check_data(self):
        # 检查数据是否符合入模标准,主要检查检查字符串列、空缺值列、无穷值列这三项。
        print('* 开始检查数据...')
        # 字符串排查
        string_cols = [col for col in self.x.columns if self.x[col].dtype == object]
        if len(string_cols):
            print('含有字符串的列有:{}'.format(','.join(string_cols)))

        # 空缺值排查
        nan_list = self.x.isnull().sum().tolist()  # 把每一列的空值个数加起来
        nan_index = [i for i, v in enumerate(nan_list) if v != 0]
        nan_cols = self.x.columns[nan_index]
        if len(nan_cols):
            print('含有空缺值的列有:{}'.format(','.join(nan_cols)))

        # 无穷值排查（正负无穷都包含）
        # 进行无穷值排查时，需要去除字符串的列
        safe_cols = list(set(self.x.columns) - set(string_cols))
        inf_list = np.isinf(self.x[safe_cols]).sum().tolist()  # 把每一列的无穷值个数加起来
        inf_index = [i for i, v in enumerate(inf_list) if v != 0]
        inf_cols = self.x[safe_cols].columns[inf_index]
        if len(inf_cols):
            print('含有无穷值的列有:{}'.format(','.join(inf_cols)))
        in_model = 1 if len(string_cols) or len(nan_cols) or len(inf_cols) else 0
        return in_model

    def calc_model_metrics(self, clf):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test

        # train scores
        train_accuracy = accuracy_score(y_train, clf.predict(x_train))
        train_precision = precision_score(y_train, clf.predict(x_train))
        train_recall = recall_score(y_train, clf.predict(x_train))
        train_f1 = f1_score(y_train, clf.predict(x_train))
        train_auc = roc_auc_score(y_train, clf.predict(x_train))

        # test_score
        test_accuracy = accuracy_score(y_test, clf.predict(x_test))
        test_precision = precision_score(y_test, clf.predict(x_test))
        test_recall = recall_score(y_test, clf.predict(x_test))
        test_f1 = f1_score(y_test, clf.predict(x_test))
        test_auc = roc_auc_score(y_test, clf.predict(x_test))

        all_scores = {
            "train_scores": {
                'train_accuracy': train_accuracy,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'train_auc': train_auc

            },
            "test_scores": {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc

            }
        }
        return all_scores
        # accuracy = cross_val_score(clf, self.x_test, self.y_test, cv=config.cv, scoring='accuracy').mean()
        # precision = cross_val_score(clf, self.x_test, self.y_test, cv=config.cv, scoring='precision').mean()
        # recall = cross_val_score(clf, self.x_test, self.y_test, cv=config.cv, scoring='recall').mean()
        # f1 = cross_val_score(clf, self.x_test, self.y_test, cv=config.cv, scoring='f1').mean()
        # auc = cross_val_score(clf, self.x_test, self.y_test, cv=config.cv, scoring='roc_auc').mean()
        # all_avg_scores = {'accuracy': accuracy,
        #                   'precision': precision,
        #                   'recall': recall,
        #                   'f1': f1,
        #                   'auc': auc}
        # return all_avg_scores

    @fn_timer
    def universal_model_training(self, model_name=None):

        if model_name == 'lr':
            estimator = LogisticRegression()
            param_grid = config.lr_gs_cv_hyper_params
        elif model_name == 'dt':
            estimator = DecisionTreeClassifier()
            param_grid = config.dt_gs_cv_hyper_params
        elif model_name == 'rf':
            estimator = RandomForestClassifier()
            param_grid = config.rf_gs_cv_hyper_params
        elif model_name == 'gbdt':
            estimator = GradientBoostingClassifier()
            param_grid = config.rf_gs_cv_hyper_params
        elif model_name == 'xgb':
            estimator = XGBClassifier()
            param_grid = config.xgboost_gs_cv_hyper_params
        else:
            print("暂时不支持此种算法，目前支持的算法有'lr'、'dt'、'rf'、'gbdt'、'xgb'！")
            sys.exit(1)

        model_gscv = GridSearchCV(estimator=estimator,
                                  param_grid=param_grid,
                                  scoring=config.scoring,
                                  cv=config.cv,
                                  n_jobs=-1,
                                  verbose=1,

                                  )
        model_gscv.fit(self.x_train, self.y_train)
        # 使用最优超参数进行训练

        # estimator.kwargs = model_gscv.best_params_
        # print(model_gscv.best_params_)
        estimator = model_gscv.best_estimator_
        # estimator.fit(self.x_train, self.y_train) # 这个就多此一举了，不过想看训练数据集上的效果可以打开注释
        metrics = self.calc_model_metrics(estimator)
        train_scores, test_scores = metrics['train_scores'], metrics['test_scores']

        result = {'model_name': model_name,
                  'tuned_model': estimator,
                  'train_metrics':
                      {
                          'Train_Accuracy': train_scores['train_accuracy'],
                          'Train_Precision': train_scores['train_precision'],
                          'Train_Recall': train_scores['train_recall'],
                          'Train_F1': train_scores['train_f1'],
                          'Train_AUC': train_scores['train_auc'],
                      },
                  'test_metrics':
                      {
                          'Test_Accuracy': test_scores['test_accuracy'],
                          'Test_Precision': test_scores['test_precision'],
                          'Test_Recall': test_scores['test_recall'],
                          'Test_F1': test_scores['test_f1'],
                          'Test_AUC': test_scores['test_auc'],
                      },
                  }
        pp(result)

        # TODO 绘图类还有待优化 主要是roc和pr曲线
        pt = PlottingTool(estimator, self.x_test, self.y_test)
        pt.plot_roc()
        pt.plot_pr()
        pt.plot_confusion_matrix()

    def imblearn_classification(self, model_name=None):
        if model_name == 'brc':
            estimator = brc()
            param_grid = config.brc_gs_cv_hyper_params
        elif model_name == 'bbc':
            estimator = bbc()
            param_grid = config.bbc_gs_cv_hyper_params
        elif model_name == 'rusbc':
            estimator = rusbc()
            param_grid = config.rusbc_gs_cv_hyper_params
        elif model_name == 'eec':
            estimator = eec()
            param_grid = config.eec_gs_cv_hyper_params
        else:
            print("非平衡不支持此种算法，请检查！")
            sys.exit(1)

        model_gscv = GridSearchCV(estimator=estimator,
                                  param_grid=param_grid,
                                  scoring=config.scoring,
                                  cv=config.cv,
                                  n_jobs=-1,
                                  verbose=1,

                                  )
        model_gscv.fit(self.x_train, self.y_train)
        # 使用最优超参数进行训练

        print("最优的超参数为：", model_gscv.best_params_)
        estimator = model_gscv.best_estimator_
        # estimator.fit(self.x_train, self.y_train) # 这个就多此一举了，不过想看训练数据集上的效果可以打开注释
        metrics = self.calc_model_metrics(estimator)
        train_scores, test_scores = metrics['train_scores'], metrics['test_scores']

        result = {'model_name': model_name,
                  'tuned_model': estimator,
                  'train_metrics':
                      {
                          'Train_Accuracy': train_scores['train_accuracy'],
                          'Train_Precision': train_scores['train_precision'],
                          'Train_Recall': train_scores['train_recall'],
                          'Train_F1': train_scores['train_f1'],
                          'Train_AUC': train_scores['train_auc'],
                      },
                  'test_metrics':
                      {
                          'Test_Accuracy': test_scores['test_accuracy'],
                          'Test_Precision': test_scores['test_precision'],
                          'Test_Recall': test_scores['test_recall'],
                          'Test_F1': test_scores['test_f1'],
                          'Test_AUC': test_scores['test_auc'],
                      },
                  }
        pp(result)

        pt = PlottingTool(estimator, self.x_test, self.y_test)
        pt.plot_roc()
        pt.plot_pr()
        pt.plot_confusion_matrix()
        return result

    def feature_importance(self):
        pass


if __name__ == '__main__':
    print("Hello World!")
