# 归一化
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def minmax_demo():
    """
    归一化演示
    :return: None
    """
    data = pd.read_csv("data/dating.txt")
    print(data)
    # 1、实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(2, 3))
    # 2、调用fit_transform
    data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
    print("最小值最大值归一化处理的结果：\n", data)

    return None


# 标准化
import pandas as pd
from sklearn.preprocessing import StandardScaler

def stand_demo():
    """
    标准化演示
    :return: None
    """
    data = pd.read_csv("data/dating.txt")
    print(data)
    # 1、实例化一个转换器类
    transfer = StandardScaler()
    # 2、调用fit_transform
    data = transfer.fit_transform(data[['milage','Liters','Consumtime']])
    print("标准化的结果:\n", data)
    print("每一列特征的平均值：\n", transfer.mean_)
    print("每一列特征的方差：\n", transfer.var_)

    return None

# 归一化实例化
minmax_demo()
print("---------------------")
# 标准化实例化
stand_demo()
print("---------------------")


#K近邻算法的API
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()  # 加载数据集
x_ = iris.data     # 获取数据集特征值
y_ = iris.target   # 获取数据集目标值

estimator = KNeighborsClassifier(n_neighbors=3)  # 创建算法模型对象	（K的取值通过n_neighbors传递#，）
estimator.fit(x_, y_)  # 调用fit方法训练模型
predictions = estimator.predict(x_)           # 用训练好的模型进行预测
print(predictions) #打印
print("---------------------")



##分类算法的评估
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#加载鸢尾花数据
X,y = datasets.load_iris(return_X_y = True)
#训练集 测试集划分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# 创建KNN分类器对象 近邻数为6
knn_clf = KNeighborsClassifier(n_neighbors=6)
#训练集训练模型
knn_clf.fit(X_train,y_train)
#使用训练好的模型进行预测
y_predict = knn_clf.predict(X_test)

# 计算准确率：
print(sum(y_predict==y_test)/y_test.shape[0])
print("---------------------")


#API计算准确率
from sklearn.metrics import accuracy_score
#方式1：
print(accuracy_score(y_test,y_predict))
# #方式2：
# knn_classifier.score(X_test,y_test)

