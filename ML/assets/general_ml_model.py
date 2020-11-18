'''
create by xubing on 2020-03-26
一种通用的机器学习模型

最重要的一些参数：
max_depth：树的最大深度
num_leaves：叶子节点个数
min_data_in_leaf：落在叶子节点上的最小数据数
bagging_fraction：bagging时的比例
bagging_freq：bagging时的频率
feature_fraction：特征选择的比例
max_bin：最大桶的个数
learning_rate：学习率
num_iterations：迭代次数
由于参数比较多，如果直接用GridSearchCV会导致组合数迅速膨胀（连乘）
因此，针对不同的目标，固定一些重要的参数，其他的参数进行GridSearch可能会好一些
因此，有以下四种超参数搜索方式。
leaf-wise
high-speed 更快的速度
high-accuracy 更高的准确率
avoid-over-fitting 避免过拟合
'''

import warnings

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as sp
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
import time


def get_pretty_time(seconds):
    if seconds == -1:
        return 'N/A'

    if seconds <= 60:
        return '%d秒' % seconds

    if seconds <= 3600:
        return '%d分钟%d秒' % (seconds // 60, seconds % 60)

    return '%d小时%d分' % (seconds // 3600, seconds % 3600 // 60)


class Config(object):
    def __init__(self):
        self.model_config = {
            'boosting_type': 'gbdt',  # rf dart goss
            'objective': 'binary',  # 任务 可选 {binary（二分类）、multi（多分类）、regression（回归）}
            'data_scale': 'middel_small',  # 数据规模
            'test_size': 0.2,
            'num_class': 3,
            'metric': 'f1',  # 可选的参数参见 sklearn.metrics.SCORES.keys()
            'folds': 4,
            'verbose': 1  # 值越大，显示的log越多 1 5 10 50
        }
        self.all_params = {
            'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
            'n_estimators': [10, 100, 1000],
            'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],

        }
        # 如果用grid search  cv  42 5250 0000 * 4 = 170亿1千万个组合
        self.all_params = {
            # Step 1 超参数
            # #max_depth:树深度。 越大，越准确，越慢，越可能过拟合。
            # #num_leaves: 叶子节点个数 。越大，越准确，越慢、越可能过拟合
            # #max_bin:工具箱数。（叶子节点+非叶子节点） 越大，越准确，越慢，越可能过拟合。
            'max_depth': [3, 5, 7, 9, 12, 15, 17, 25, -1],  # 默认 -1
            'num_leaves': [15, 31, 63, 127, 255],  # 默认 127
            'max_bin': [63, 127, 255],  # 默认 255

            # Step2 超参数
            # #feature_fraction:随机选择的特征比例。 越大，越准确，越慢，越可能过拟合。
            # #bagging_fraction：随机选择的数据比例。 越大，越准确，越慢，越可能过拟合。
            # #bagging_freq：随机选择数据的频率。越小，越准确，越慢，越可能过拟合。
            'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],  # 默认 1
            'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],  # 默认 1
            'bagging_freq': range(0, 81, 10),  # 默认0 表示禁用

            # Step 3 超参数
            # # lambda_l1：L1正则。越小，越准确，越慢，越可能过拟合。
            # # lambda_l2：L2正则。越小，越准确，越慢，越可能过拟合。
            'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # 默认 0
            'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0],  # 默认 0

            # Step4 超参数
            # # min_split_gain:执行切分的最小增益。
            # #min_data_in_leaf:落在叶子节点的最小数据数量。越小，越准确，越慢，越可能过拟合。
            'min_split_gain': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0],  # 默认 0
            'min_data_in_leaf': [10, 20, 50, 100, 200],  # 默认 100

            # Step5 超参数
            # #learning_rate:学习率（步长）。越小，越准确，越慢，越可能过拟合。
            # #num_iterations:迭代次数。（alias = num_trees, num_rounds）越大，越准确，越慢，越可能过拟合。
            'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 1],  # 默认 0.1
            'num_iterations': [100, 500, 1000, 2000]  # 默认 100
        }



class GeneralMLModel:

    def __init__(self, X, y):
        config = Config().model_config
        self.boosting_type = config['boosting_type']
        self.objective = config['objective']
        self.data_scale = config['data_scale']  # 数据规模 可选{middle_small(middle and small中小规模)、large(大规模)}
        self.test_size = config['test_size']
        self.metric = config['metric']
        self.num_class = config['num_class']  # 多分类可用
        self.folds = config['folds']
        self.verbose = config['verbose']

        self.all_params = Config().all_params
        self.trainX, self.testX, self.trainY, self.testY = sp(X, y, test_size=self.test_size)

    def lr_reg(self):
        reg = LinearRegression().fit(self.trainX, self.trainY)
        return reg.score(self.testX, self.testY)

    def lr_clf(self):
        lr_start_time = time.time()
        clf = LogisticRegression().fit(self.trainX, self.trainY)
        lr_clf_end_time = time.time()
        lr_clf_cost_time = lr_clf_end_time - lr_start_time
        print('lr clf fit cost time:%s' % get_pretty_time(lr_clf_cost_time))
        clf_cv = GridSearchCV(estimator=LogisticRegression(),
                              param_grid={},
                              scoring=self.metric,
                              cv=self.folds,
                              n_jobs=-1,
                              )
        clf_cv.fit(self.trainX, self.trainY)
        lr_clf_cv_cost_time = time.time() - lr_clf_end_time
        print('lr clr_cv fit cost time:%s' % get_pretty_time(lr_clf_cv_cost_time))
        return {
            'lr clf score:', f1_score(self.testY, clf.predict(self.testX)),
            'lr_cv clf score:', f1_score(self.testY, clf_cv.predict(self.testX)),
        }

    def gbdt_reg(self):
        reg = GradientBoostingRegressor().fit(self.trainX, self.trainY)
        return reg.score(self.testX, self.testY)

    def gbdt_clf(self):
        start_time = time.time()
        clf = GradientBoostingClassifier().fit(self.trainX, self.trainY)
        gbdt_clf_end_time = time.time()
        print('gbdt clf fit cost time:%s' % get_pretty_time(gbdt_clf_end_time - start_time))

        clf_cv = GridSearchCV(estimator=GradientBoostingClassifier(),
                              param_grid={},
                              scoring=self.metric,
                              cv=self.folds,
                              n_jobs=-1,
                              )
        clf_cv.fit(self.trainX, self.trainY)
        gbdt_clf_cv_end_time = time.time()
        print('gbdt clr_cv fit cost time:%s' % get_pretty_time(gbdt_clf_cv_end_time - gbdt_clf_end_time))

        # 超参数组合数
        num_conbins = 1
        for val in self.all_params.values():
            num_conbins *= len(val)
        print('超参数组合数：', num_conbins)

        clf_gscv = GridSearchCV(estimator=GradientBoostingClassifier(),
                                param_grid=self.all_params,
                                scoring=self.metric,
                                cv=self.folds,
                                n_jobs=-1,
                                )
        clf_gscv.fit(self.trainX, self.trainY)
        gbdt_clf_gscv_end_time = time.time()
        print('gbdt clf_gscv fit cost time:%s' % get_pretty_time(gbdt_clf_gscv_end_time - gbdt_clf_cv_end_time))

        # return {
        #     'gbdt clf score:', f1_score(self.testY, clf.predict(self.testX)),
        #     'gbdt_cv clf score:', f1_score(self.testY, clf_cv.predict(self.testX)),
        #     'gbdt_gscv clf score:', f1_score(self.testY, clf_gscv.predict(self.testX)),
        # }

    def xgboost_clf(self):
        xgb = XGBClassifier()
        start_time = time.time()
        xgboost_clf = xgb.fit(self.trainX, self.trainY)
        xgboost_clf_end_time = time.time()
        print('xgboost clf cost time:', get_pretty_time(xgboost_clf_end_time - start_time))

        xgboost_cv_clf = GridSearchCV(
            estimator=xgb,
            param_grid={},
            scoring=self.metric,
            cv=self.folds,
            n_jobs=-1,
        )
        xgboost_cv_clf.fit(self.trainX, self.trainY)
        xgboost_cv_clf_end_time = time.time()
        print('xgboost clf cv cost time:', get_pretty_time(xgboost_cv_clf_end_time - xgboost_clf_end_time))

        xgboost_gscv_clf = GridSearchCV(
            estimator=xgb,
            param_grid=self.all_params,
            scoring=self.metric,
            cv=self.folds,
            n_jobs=-1,
        )
        xgboost_gscv_clf.fit(self.trainX, self.trainY)
        xgboost_gscv_clf_end_time = time.time()
        print('xgboost clf gscv cost time:', get_pretty_time(xgboost_gscv_clf_end_time - xgboost_cv_clf_end_time))

    def my_base_line(self):
        base_line = {
            # 'Linear Regressor R2': self.lr_reg(),
            # 'GBDT Regressor R2': self.gbdt_reg(),
            'Logistic Classifier F1': self.lr_clf(),
            # 'GBDT Classifier F1': self.gbdt_clf(),
            # 'XGBoost Classifier F1': self.xgboost_clf(),
        }
        # for v in base_line.values():
        #     for k_, v_ in v.items():
        #         print(k_, ':', v_)
    def my_metrics(self):
        pass


    def auto_tune_gbdt(self):
        # 判断机器学习任务
        # 三种常用算法
        if self.objective == 'binary':
            objective = 'binary'
        elif self.objective == 'multiclass':
            objective = 'multiclass'
            num_class = self.num_class
        elif self.objective == 'multiclassova':
            objective = 'multiclassova'
            num_class = self.num_class
        elif self.objective == 'regresssion':
            objective = 'regression'
            metric = 'l2'
        start_time = time.time()
        print('=====改善后======')
        lgb_train = lgb.Dataset(self.trainX, self.trainY)
        lgb_eval = lgb.Dataset(self.testX, self.testY, reference=lgb_train, free_raw_data=False)

        fixed_params = {
            'boosting_type': self.boosting_type,
            'objective': self.objective,
            'metric': self.metric,
            'save_binary': True,
            'random_state': 43,
        }

        print('======Step1 Start...======')
        # Step1 提高精度 max_depth、num_leaves、max_bin
        step1_params = {
            'max_depth': self.all_params['max_depth'],
            'num_leaves': self.all_params['num_leaves'],
            'max_bin': self.all_params['max_bin']
        }
        gs1 = GridSearchCV(estimator=lgb.LGBMClassifier(**fixed_params),
                           param_grid=step1_params,
                           scoring=self.metric,
                           cv=self.folds,
                           n_jobs=4,
                           verbose=self.verbose
                           )

        gs1.fit(self.testX, self.testY)
        print("======Step1 End and Step1's Result======")
        print("Best params:", gs1.best_params_)
        print("Best {0} score:{1}".format(self.metric, gs1.best_score_))
        step1_end_time = time.time()
        step1_cost_time = step1_end_time - start_time
        print('Step1 花费时间 %s' % get_pretty_time(step1_cost_time))

        # 更新fixed_params
        fixed_params['max_depth'] = gs1.best_params_['max_depth']
        fixed_params['num_leaves'] = gs1.best_params_['num_leaves']

        print('======Step2 Start...======')
        # Step2 降低过拟合 feature_fraction bagging_fraction bagging_freq
        step2_params = {
            'feature_fraction': self.all_params['feature_fraction'],
            'bagging_fraction': self.all_params['bagging_fraction'],
            'bagging_freq': self.all_params['bagging_freq']
        }
        gs2 = GridSearchCV(estimator=lgb.LGBMClassifier(**fixed_params),
                           param_grid=step2_params,
                           scoring=self.metric,
                           cv=self.folds,
                           n_jobs=-1,
                           verbose=self.verbose
                           )
        gs2.fit(self.trainX, self.trainY)
        print('====== Step2 Result======')
        print("Best params:", gs2.best_params_)
        print("Best {0} score:{1}".format(self.metric, gs2.best_score_))
        step2_end_time = time.time()
        step2_cost_time = step2_end_time - step1_end_time
        print('Step2 花费时间 %s' % get_pretty_time(step2_cost_time))

        # 更新fixed_params
        fixed_params['feature_fraction'] = gs2.best_params_['feature_fraction']
        fixed_params['bagging_fraction'] = gs2.best_params_['bagging_fraction']
        fixed_params['bagging_freq'] = gs2.best_params_['bagging_freq']

        print('======Step3 Start...======')
        # Step3 降低过拟合 lambda_l1 lambda_l2
        step3_params = {
            'lambda_l1': self.all_params['lambda_l1'],
            'lambda_l2': self.all_params['lambda_l2'],
        }
        gs3 = GridSearchCV(estimator=lgb.LGBMClassifier(**fixed_params),
                           param_grid=step3_params,
                           scoring=self.metric,
                           cv=self.folds,
                           n_jobs=-1,
                           verbose=self.verbose
                           )
        gs3.fit(self.trainX, self.trainY)
        print('====== Step3 Result======')
        print("Best params:", gs3.best_params_)
        print("Best {0} score:{1}".format(self.metric, gs3.best_score_))
        step3_end_time = time.time()
        step3_cost_time = step3_end_time - step2_end_time
        print('Step3 花费时间 %s' % get_pretty_time(step3_cost_time))

        # 更新fixed_params
        fixed_params['lambda_l1'] = gs3.best_params_['lambda_l1']
        fixed_params['lambda_l2'] = gs3.best_params_['lambda_l2']

        print('======Step4 Start...======')
        # Step4 提升精度+降低过拟合 lambda_l1、lambda_l2
        step4_params = {
            'min_split_gain': self.all_params['min_split_gain'],
            'min_data_in_leaf': self.all_params['min_data_in_leaf'],
        }
        gs4 = GridSearchCV(estimator=lgb.LGBMClassifier(**fixed_params),
                           param_grid=step4_params,
                           scoring=self.metric,
                           cv=self.folds,
                           n_jobs=-1,
                           verbose=self.verbose
                           )
        gs4.fit(self.trainX, self.trainY)
        print('====== Step4 Result======')
        print("Best params:", gs4.best_params_)
        print("Best {0} score:{1}".format(self.metric, gs4.best_score_))
        step4_end_time = time.time()
        step4_cost_time = step4_end_time - step3_end_time
        print('Step4 花费时间 %s' % get_pretty_time(step4_cost_time))

        # 更新fixed_params
        fixed_params['min_split_gain'] = gs4.best_params_['min_split_gain']
        fixed_params['min_data_in_leaf'] = gs4.best_params_['min_data_in_leaf']

        print('======Step5 Start...======')
        # Step5 提升精度 learning_rate、num_rounds
        step5_params = {
            'learning_rate': self.all_params['learning_rate'],
            'num_iterations': self.all_params['num_iterations'],
        }
        gs5 = GridSearchCV(
            estimator=lgb.LGBMClassifier(**fixed_params),
            param_grid=step5_params,
            scoring=self.metric,
            cv=self.folds,
            n_jobs=-1,
            verbose=self.verbose
        )
        gs5.fit(self.trainX, self.trainY)
        print('====== Step5 Result======')
        print("Best params:", gs5.best_params_)
        print("Best {0} score:{1}".format(self.metric, gs5.best_score_))
        step5_end_time = time.time()
        step5_cost_time = step5_end_time - step4_end_time
        print('Step5 花费时间 %s' % get_pretty_time(step5_cost_time))

        # 更新fixed_params
        fixed_params['learning_rate'] = gs5.best_params_['learning_rate']
        fixed_params['num_iterations'] = gs5.best_params_['num_iterations']

        all_end_time = time.time()
        all_cost_time = all_end_time - start_time
        print('5 steps 花费时间:%s' % get_pretty_time(all_cost_time))

        print('==========最优超参数如下==========')
        for k, v in fixed_params.items():
            print(k, ':', v)

        print('=======使用默认超参数训练模型...========')
        model = lgb.LGBMClassifier()
        model.fit(self.trainX, self.trainY)
        if self.metric == 'f1':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))
        elif self.metric == 'recall':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))
        elif self.metric == 'precision':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))



        print('=======使用最优超参数训练模型...========')
        model = lgb.LGBMClassifier(**fixed_params)
        model.fit(self.trainX, self.trainY)
        if self.metric == 'f1':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))
        elif self.metric == 'recall':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))
        elif self.metric == 'precision':
            print('recall score:', recall_score(self.testY, model.predict(self.testX)))
            print('precision score:', precision_score(self.testY, model.predict(self.testX)))
            print('f1 score:', f1_score(self.testY, model.predict(self.testX)))


