import pandas as pd
from sklearn.model_selection import train_test_split as sp
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

#==========配置==========
file = 'iris_2.csv'#文件名
target = 'class'#目标变量
test_size = 0.2#测试集占比

df = pd.read_csv(file)
X = df.drop(target, axis=1)
y = df[target]

trainX, testX, trainY, testY = sp(X, y, test_size=test_size)
clf = lr()
clf.fit(trainX, trainY)
print('=========================')
print('Base_line的Acc是：%.4f\t|'%clf.score(testX, testY))
print('Base_line的Auc是：%.4f\t|'% roc_auc_score(testY, clf.predict(testX)))
print('=========================')

