"""
create by xubing 2020-09-15

我的常用绘图类
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import auc
from inspect import signature
import scikitplot as skplt


import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Heiti'


class FigurePlotter:
    def __init__(self):
        self.name = 'fig_plot'

    # 条形图
    def plot_barh(self, key: list, value: list, order=None):
        """
        水平条形图(一般用于排序)
        """
        if order is not None:
            pass
        sns.barplot(x=key, y=value, order=order)
        plt.show()

    def plot_barv(self, key: list, value: list, order=None):
        '''
        竖直条形图
        '''
        ...

    def plot_barv_multi(self, key: list, value: list, order=None):
        '''
        多条竖直条形图
        '''
        ...

    # 折线图
    def plot_line(self):
        ...

    # 饼图
    def plot_pie(self):
        ...

    # ML Metric Score
    def plot_confusion_matrix(self, estimator, x_test, y_test, save_path=''):
        # Method 1
        # f, ax = plt.subplots()
        # y_pred = estimator.predict(x_test)

        # # 多分类也是可以的
        # # y_test = [0, 1, 2, 3, 2, 3, 1, 2, 3, 1, 2]
        # # y_pred = [1, 2, 1, 2, 1, 1, 3, 1, 2, 2, 2]
        # sns.heatmap(confusion_matrix(y_test, y_pred),
        #             annot=True, fmt='g', cmap='GnBu')

        # ax.set_title('Confusion matrix')  # 标题
        # ax.set_xlabel('Predict')  # x轴
        # ax.set_ylabel('True')  # y轴

        # Method 2
        y_pred = estimator.predict(x_test)
        skplt.metrics.plot_confusion_matrix(
            y_test, y_pred, normalize=True, cmap='GnBu')
        plt.show()
        # plt.savefig(save_path)

    def plot_pr_curve(self, estimator, x_test, y_test, save_path=''):
        probas = estimator.predict_proba(x_test)
        skplt.metrics.plot_precision_recall_curve(y_test, probas)
        plt.show()

    def plot_roc(self, estimator, x_test, y_test):
        y_probas = estimator.predict_proba(x_test)
        skplt.metrics.plot_roc(y_test, y_probas)
        plt.show()

    # Finnal Metric

    def plot_psi_csi_curve(self):
        pass

    def plot_ks_curve(self, estimator, x_test, y_test, save_path=''):
        y_probas = estimator.predict_proba(x_test)
        skplt.metrics.plot_ks_statistic(y_test, y_probas)
        plt.show()

    def plot_lift_curve(self, estimator, x_test, y_test, save_path=''):
        y_probas = estimator.predict_proba(x_test)
        skplt.metrics.plot_ks_statistic(y_test, y_probas)
        skplt.metrics.plot_lift_curve(y_test, y_probas)

    # def plot_feature_importance(self, estimator, x_test, y_test):
    #     有些算法没有feature_importance属性
    #     skplt.estimators.plot_feature_importances(estimator, x_test.columns)
    #     plt.show()
        # y_pred = estimator.predict(x_test)
        # y_pred_all = np.zeros((len(y_pred), 2), dtype=np.float32)
        # y_pred_all[:, 1] = y_pred
        # y_pred_all[:, 0] = 1 - y_pred
        # skplt.metrics.plot_cumulative_gain(y, y_pred_all)
        # plt.show()

    #     y_pred_int = y_pred.copy()
    #     y_pred_int[y_pred_int >= 0.5] = 1
    #     y_pred_int[y_pred_int < 0.5] = 0
    #
    #     # 混淆矩阵
    #     cm = confusion_matrix(y, y_pred_int)
    #     ConfusionMatrixDisplay(cm).plot()
    #     self.report_matplot('混淆矩阵')
    #
    #     # ROC曲线
    #     auc = roc_auc_score(y, y_pred)
    #     fpr, tpr, _ = roc_curve(y, y_pred, pos_label=1)
    #     disp = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(label=f'AUC={auc:.4f}')
    #     self.report_matplot('RoC曲线')
    #
    #     # PR曲线
    #     prec, recall, _ = precision_recall_curve(y, y_pred, pos_label=1)
    #     disp = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    #     self.report_matplot('Precision-Recall曲线')
    #
    #     # 下面这几类图需要正负样本两类的预测值
    #     y_pred_all = np.zeros((len(y_pred), 2), dtype=np.float32)
    #     y_pred_all[:, 1] = y_pred
    #     y_pred_all[:, 0] = 1 - y_pred
    #
    #     # cumulative gain曲线
    #     skplt.metrics.plot_cumulative_gain(y, y_pred_all)
    #     self.report_matplot('Cumulative Gain曲线')
    #
    #     # KS曲线
    #     skplt.metrics.plot_ks_statistic(y, y_pred_all)
    #     self.report_matplot('KS曲线')
    #
    #     # Lift曲线
    #     skplt.metrics.plot_lift_curve(y, y_pred_all)
    #     self.report_matplot('Lift曲线')
if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split as sp
    from sklearn.metrics import confusion_matrix
    from sklearn.linear_model import LogisticRegression as lr

    file = '/Users/xubing/DataSets/Titanic/titanic.csv'
    df = pd.read_csv(file)
    target = 'Survived'
    X = df.drop(target, axis=1)
    keep_cols = []
    for col in X.columns:
        if X[col].dtype != object:
            keep_cols.append(col)
    X = X[keep_cols]
    X.fillna(0, inplace=True)
    # X.fillna()
    y = df[target]
    x_train, x_test, y_train, y_test = sp(X, y, test_size=0.25)

    clf = lr()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    fp = FigurePlotter()
    # fp.plot_confusion_matrix(clf, x_test, y_test)
    # fp.plot_roc(clf, x_test, y_test)
    # fp.plot_pr_curve(clf, x_test, y_test)
    # fp.plot_ks_curve(clf, x_test, y_test)
    fp.plot_featur_importance(clf, x_test, y_test)
