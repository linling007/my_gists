"""
Created by xubing on 2020/6/8

"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, f1_score
import time


def get_pretty_time(seconds):
    if seconds == -1:
        return 'N/A'
    if seconds <= 60:
        return '%d秒' % seconds
    if seconds <= 3600:
        return '%d分钟%d秒' % (seconds // 60, seconds % 60)
    return '%d小时%d分' % (seconds // 3600, seconds % 3600 // 60)


# 计算运行时间的装饰器
def fn_timer(function):
    # @wraps(function)
    def function_timer(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print('======================')
        print("%s 模型运行时间为:  %s " % (args[1], get_pretty_time(end_time - start_time)))
        print('=====================')
        return result

    return function_timer

class PlottingTool:
    def __init__(self, estimator, x_test, y_test):
        self.estimator = estimator
        # self.x_train = x_train
        self.x_test = x_test
        # self.y_train = y_train
        self.y_test = y_test

    def plot_roc(self):
        # 绘制roc曲线
        estimator = self.estimator
        testX = self.x_test
        testy = self.y_test

        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(testy))]

        # predict probabilities
        lr_probs = estimator.predict_proba(testX)

        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        # calculate scores
        ns_auc = roc_auc_score(testy, ns_probs)
        lr_auc = roc_auc_score(testy, lr_probs)

        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))

        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)

        # plot the roc curve for the model
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        # show the legend
        plt.legend()

        # show the plot
        plt.show()

        # save the plot
        # plt.savefig('roc')

    def plot_pr(self):
        # 绘制pr曲线
        estimator = self.estimator

        testX = self.x_test
        testy = self.y_test

        # predict probabilities
        lr_probs = estimator.predict_proba(testX)

        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]

        # predict class values
        yhat = estimator.predict(testX)
        lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
        lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)

        # summarize scores
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        # plot the precision-recall curves
        no_skill = len(testy[testy == 1]) / len(testy)

        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    def plot_confusion_matrix(self):
        # 绘制混淆矩阵
        estimator = self.estimator

        x_test = self.x_test
        y_test = self.y_test
        y_pre = estimator.predict(x_test)
        sns.heatmap(confusion_matrix(y_test, y_pre), annot=True, fmt='2.0f', cmap='YlGn')  # Set2
        # plt.savefig(save_path+'/matrix.png', dpi=200)
        plt.title('Classification result')
        plt.xlabel('Predict')
        plt.ylabel('True')
        plt.show()

    # def plot_dt_graph(self):
    #     # 绘制决策图
    #     estimator = self.estimator
    #     x_train = self.x_train
    #     y_train = self.y_train
    #
    #     fig = plt.figure(figsize=(12, 12), dpi=80)
    #     tree.plot_tree(estimator,
    #                    feature_names=x_train.columns,
    #                    class_names=[str(i) for i in y_train.value_counts().index.tolist()],
    #                    filled=True,
    #                    rounded=True,
    #                    fontsize=None
    #                    )
    #     # plt.savefig('DTG.png')
    #     plt.show()