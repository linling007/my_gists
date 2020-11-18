"""
create by xubing 2019-10-10
绘制决策图
"""

# # %matplotlib inline #在Jupyter-Notebook内嵌图
# from sklearn.externals.six import StringIO
# import pydotplus
import graphviz
from six import StringIO
import json

import pydotplus
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
print(clf.score(iris.data, iris.target))
# fig = plt.figure(figsize=(12, 12), dpi=80)
# tree.plot_tree(clf,
#                feature_names=iris.feature_names,
#                class_names=iris.target_names,
#                filled=True,
#                rounded=True,
#                fontsize=None
#                )
# plt.savefig('DTC.png')
# plt.show()


# # ===============================================================

# # drt是DecisionTreeClassifier()，在之前要fit训练之后才能在这里输出图形。
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # graph.write_png("out.png")  # 当前文件夹生成out.png

# # 这三行代码可以生成pdf：
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render()
# with open('dt_data.csv', 'w') as f:
#     f.write(dot_data)

# json.dumps(dot_data)
print(type(dot_data))
