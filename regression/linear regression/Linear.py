

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
线性回归
"""

# 导入数据，并查看
# names添加列名，header用指定的行来作为标题，若原无标题且指定标题则设为tNone
data = pd.read_csv('./ex1data1.txt',names=['population','profit'])
#data.head(9)
#data.describe()

# 对于这个数据集，使用散点图来可视化数据，因为它只有两个属性(利润和人口)。
data.plot.scatter('population','profit',label='population')
plt.show()

# 计算代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)
    return np.sum(inner) / (2 * len(X))

# 在训练集中添加一列，以便可以使用向量化的解决方案来计算代价和梯度。
data.insert(0,'ones',1)#第零排赋值1，label为one
# data.head()

# 取最后一列为 y，其余为 X
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X = X.values
#X.shape

y = y.values
y = y.reshape(97,1)
# y.shape

# 线性回归的代价函数
def costFunction(X,y,theta):
    inner =np.power( X @ theta - y, 2)
    return np.sum(inner) / (2 * len(X))
#(∑(xi*wi)^2-y)/2*len(x)线性回归公式
# 初始化theta
theta = np.zeros((2,1))
# theta.shape//最好是先确定x的列数

# 计算初始代价函数的值 (theta初始值为0).
# cost_init = costFunction(X,y,theta)
# print(cost_init)

# 批量梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    costs = []
    
    for i in range(iters):
         # 利用向量化的矩阵求解
        theta = theta - (X.T @ (X@theta - y) ) * alpha / len(X)  # 求出迭代后的theta
        cost = costFunction(X,y,theta)
        costs.append(cost)
        
        if i % 100 == 0:
            print(cost)
            
    return theta,costs


alpha = 0.02
iters = 2000

# 初始化学习速率和迭代次数
# theta为迭代完之后的theta
theta,costs = gradientDescent(X,y,theta,alpha,iters)


# 代价函数的曲线
fig,ax = plt.subplots()
ax.plot(np.arange(iters),costs)   # np.arange()返回等差数组
ax.set(xlabel='iters',
      ylabel='cost',
      title='cost vs iters')
plt.show()


# 绘制线性模型以及数据，直观地看出它的拟合。
x = np.linspace(y.min(),y.max(),100)
y_ = theta[0,0] + theta[1,0] * x


fig,ax = plt.subplots()
ax.scatter(X[:,1],y,label='training data')
ax.plot(x,y_,'r',label='predict')
ax.legend()
ax.set(xlabel='populaiton',
      ylabel='profit')
plt.show()
