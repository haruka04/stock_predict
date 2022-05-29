import numpy as np
import math
import matplotlib.pyplot as pyplot
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score
# グリッドサーチCVの導入
from sklearn.model_selection import GridSearchCV
# 学習データと評価データへ分割するライブラリの導入
from sklearn.model_selection import train_test_split


data = np.loadtxt('./stock_softbank_data/stock_data_caseE.csv', delimiter=',', dtype='float')
y = data[:,0].astype(int)
x = data[:,1]
x = x.reshape(-1,1)
# print('y')
# # print(y)
# # print('x')
# # print(x)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

data_test = np.loadtxt('./stock_softbank_data/stock_data_test.csv', delimiter=',') 
test_y = data_test[:,0].astype(int)
test_x = data_test[:,1]
test_x = test_x.reshape(-1,1)

print('正解',test_y)

# 決定技分類
# decision_param_grid = {"max_depth": [1, 2, 3, 4, 5],
#                        "min_samples_leaf": [2, 3, 4, 5]}

# clf = GridSearchCV(tree.DecisionTreeClassifier(), decision_param_grid, cv=5, scoring='precision')
# clf.fit(x,y)
# print("最良のパラメータ")
# print(clf.best_params_)
# print('Best cross-validation: {}'.format(clf.best_score_))

# # 結果
# print('決定技')
# print('決定技が推測した結果',clf.predict(test_x))
# print('決定技が推測した結果の正答率',clf.score(test_x, test_y))

# ロジスティック回帰分類
# param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
# clf = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='precision')
# clf.fit(x,y)
# print('最良のパラメータ')
# print(clf.best_params_)
# print('Best cross-validation: {}'.format(clf.best_score_))

# 結果
# print('ロジステック回帰')
# print('ロジステックが推測した結果',clf.predict(test_x))
# print('ロジステックが推測した結果の正答率',clf.score(test_x, test_y))
# print('決定木')
# print('caseE')
# # グリッドサーチしたいパラメータCの値域をnp.logspaceで設定
param_grid_svm = {'C': [0.000001,0.00001,0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000], 'gamma' : [0.000001, 0.00001, 0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000,10000,100000]}
# param_grid_svm = {'C': [ 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma' : [ 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

param_grid_tree = {"max_depth": [1, 2, 3, 4, 5],
                       "min_samples_leaf": [2, 3, 4, 5]}

param_grid_logi = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],"random_state":[123]}

param_grid_k = {"n_neighbors":[1, 3, 5, 7, 9, 11, 15, 21],"weights":["uniform","distance"]}
# 各モデルを一つずつCに関してグリッドサーチを行う
print('caseE')
# SVM
clf = GridSearchCV(svm.SVC(), param_grid_svm, cv=10, scoring='precision')
clf.fit(x,y)
print('最良のパラメータ')
print(clf.best_params_)
print('Best cross-validation: {}'.format(clf.best_score_))

# 学習したデータと比較して推測する
# print('SVM')
# print('SVMが推測した結果',clf.best_estimator_.predict(test_x))
print('SVMが推測した結果の正解率',clf.score(test_x, test_y))

# 決定木
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid_tree, cv=5, scoring='precision')
clf.fit(x,y)
print('決定木が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////

# ロジスティック回帰
clf = GridSearchCV(LogisticRegression(), param_grid_logi, cv=5, scoring='precision')
clf.fit(x,y)
print('ロジスティック回帰が推測した結果の正解率',clf.score(test_x, test_y))

# k近傍法
clf = GridSearchCV(KNeighborsClassifier(),param_grid_k, cv=10, scoring='precision')
clf.fit(x,y)
print('k近傍法が推測した結果の正解率',clf.score(test_x, test_y))

# SVM caseF
print('caseF')
data = np.loadtxt('./stock_softbank_data/stock_data_caseF.csv', delimiter=',', dtype='float')
y = data[:,0].astype(int)
x = data[:,1]
x = x.reshape(-1,1)

data_test = np.loadtxt('./stock_softbank_data/stock_data_test_caseF.csv', delimiter=',') 
test_y = data_test[:,0].astype(int)
test_x = data_test[:,1]
test_x = test_x.reshape(-1,1)

# 各モデルを一つずつCに関してグリッドサーチを行う
clf = GridSearchCV(svm.SVC(), param_grid_svm, cv=10, scoring='precision')
clf.fit(x,y)
print('最良のパラメータ')
print(clf.best_params_)
print('Best cross-validation: {}'.format(clf.best_score_))

#  学習したデータと比較して推測する
# print('SVM')
# print('SVMが推測した結果',clf.best_estimator_.predict(test_x))
print('SVMが推測した結果の正解率',clf.score(test_x, test_y))

# 決定木
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid_tree, cv=5, scoring='precision')
clf.fit(x,y)
print('決定木が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////

# ロジスティック回帰
clf = GridSearchCV(LogisticRegression(), param_grid_logi, cv=5, scoring='precision')
clf.fit(x,y)
print('ロジスティック回帰が推測した結果の正解率',clf.score(test_x, test_y))

# k近傍法
clf = GridSearchCV(KNeighborsClassifier(),param_grid_k, cv=10, scoring='precision')
clf.fit(x,y)
print('k近傍法が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////
# SVM caseG
print('caseG')
data = np.loadtxt('./stock_softbank_data/stock_data_caseG.csv', delimiter=',', dtype='float')
y = data[:,0].astype(int)
x = data[:,1]
x = x.reshape(-1,1)

data_test = np.loadtxt('./stock_softbank_data/stock_data_test_caseG.csv', delimiter=',') 
test_y = data_test[:,0].astype(int)
test_x = data_test[:,1]
test_x = test_x.reshape(-1,1)

# 各モデルを一つずつCに関してグリッドサーチを行う
clf = GridSearchCV(svm.SVC(), param_grid_svm, cv=10, scoring='precision')
clf.fit(x,y)
print('最良のパラメータ')
print(clf.best_params_)
print('Best cross-validation: {}'.format(clf.best_score_))

# 学習したデータと比較して推測する
# print('SVMが推測した結果',clf.best_estimator_.predict(test_x))
print('SVMが推測した結果の正解率',clf.score(test_x, test_y))

# 決定木
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid_tree, cv=5, scoring='precision')
clf.fit(x,y)
print('決定木が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////

# ロジスティック回帰
clf = GridSearchCV(LogisticRegression(), param_grid_logi, cv=5, scoring='precision')
clf.fit(x,y)
print('ロジスティック回帰が推測した結果の正解率',clf.score(test_x, test_y))

# k近傍法
clf = GridSearchCV(KNeighborsClassifier(),param_grid_k, cv=10, scoring='precision')
clf.fit(x,y)
print('k近傍法が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////
# SVM caseH
print('caseH')
data = np.loadtxt('./stock_softbank_data/stock_data_caseH.csv', delimiter=',', dtype='float')
y = data[:,0].astype(int)
x = data[:,1]
x = x.reshape(-1,1)

data_test = np.loadtxt('./stock_softbank_data/stock_data_test_caseH.csv', delimiter=',') 
test_y = data_test[:,0].astype(int)
test_x = data_test[:,1]
test_x = test_x.reshape(-1,1)

# 各モデルを一つずつCに関してグリッドサーチを行う
clf = GridSearchCV(svm.SVC(), param_grid_svm, cv=10, scoring='precision')
clf.fit(x,y)
print('最良のパラメータ')
print(clf.best_params_)
print('Best cross-validation: {}'.format(clf.best_score_))

# # 学習したデータと比較して推測する
# print('SVMが推測した結果',clf.best_estimator_.predict(test_x))
print('SVMが推測した結果の正解率',clf.score(test_x, test_y))

# 決定木
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid_tree, cv=5, scoring='precision')
clf.fit(x,y)
print('決定木が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////

# ロジスティック回帰
clf = GridSearchCV(LogisticRegression(), param_grid_logi, cv=5, scoring='precision')
clf.fit(x,y)
print('ロジスティック回帰が推測した結果の正解率',clf.score(test_x, test_y))

# k近傍法
clf = GridSearchCV(KNeighborsClassifier(),param_grid_k, cv=10, scoring='precision')
clf.fit(x,y)
print('k近傍法が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////
# SVM caseI
print('caseI')
data = np.loadtxt('./stock_softbank_data/stock_data_caseI.csv', delimiter=',', dtype='float')
y = data[:,0].astype(int)
x = data[:,1]
x = x.reshape(-1,1)

data_test = np.loadtxt('./stock_softbank_data/stock_data_test_caseI.csv', delimiter=',') 
test_y = data_test[:,0].astype(int)
test_x = data_test[:,1]
test_x = test_x.reshape(-1,1)

# 各モデルを一つずつCに関してグリッドサーチを行う
clf = GridSearchCV(svm.SVC(), param_grid_svm, cv=10, scoring='precision')
clf.fit(x,y)
print('最良のパラメータ')
print(clf.best_params_)
print('Best cross-validation: {}'.format(clf.best_score_))

# # 学習したデータと比較して推測する
# print('SVMが推測した結果',clf.best_estimator_.predict(test_x))
print('SVMが推測した結果の正解率',clf.score(test_x, test_y))

# 決定木
clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid_tree, cv=5, scoring='precision')
clf.fit(x,y)
print('決定木が推測した結果の正解率',clf.score(test_x, test_y))
# ////////////////////////////////

# ロジスティック回帰
clf = GridSearchCV(LogisticRegression(), param_grid_logi, cv=5, scoring='precision')
clf.fit(x,y)
print('ロジスティック回帰が推測した結果の正解率',clf.score(test_x, test_y))

# k近傍法
clf = GridSearchCV(KNeighborsClassifier(),param_grid_k, cv=10, scoring='precision')
clf.fit(x,y)
print('k近傍法が推測した結果の正解率',clf.score(test_x, test_y))
# pyplot.style.use('ggplot')

# x_bind = np.vstack((test_x,x))
# y_bind = np.hstack((test_y,y))

# plot_decision_regions(x_bind, y_bind, clf=clf,  res=0.02)
# pyplot.show()