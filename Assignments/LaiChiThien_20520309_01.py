import pandas as pd
from sklearn import datasets, metrics
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import seaborn as sb
from mpl_toolkits import mplot3d
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 


def normalize(x):
    return x/x.max(axis=0)


data, target = datasets.load_iris(return_X_y=True, as_frame=True)

X_train, X_test, Y_train, Y_test = train_test_split(data, target, train_size=0.7, random_state=1)

X_train = normalize(X_train)
X_test = normalize(X_test)

print(type(data))
print(data)
print(type(target))
print(target)

#Tìm các đặc trưng tương quan, để trực quan hoá được hiệu quả

dataplot = sb.heatmap(X_train.corr(), cmap="YlGnBu", annot=True)
plt.show() 

def scatter2d(X, Y, x, y):
    plt.scatter(X.iloc[:, x], X[:, y], c=Y)

plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 2], c=Y_train)
plt.show()

plt.scatter(X_train.iloc[:, 3], X_train.iloc[:, 2], c=Y_train)
plt.show()

plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 3], c=Y_train)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
x = X_train.iloc[:, 0]
y = X_train.iloc[:, 2]
z = X_train.iloc[:, 3]
ax.scatter3D(x, y, z, c=Y_train)
plt.show()

#Xử dụng KNNs có sẵn của sklearn để predict và đánh giá
model = KNeighborsClassifier(3)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print('Accuracy KNNs predict: ', metrics.accuracy_score(Y_test, Y_pred))

fig = plt.figure()
ax = plt.axes(projection='3d')
y = X_test.iloc[:, 2]
x = X_test.iloc[:, 0]
z = X_test.iloc[:, 3]
ax.scatter3D(x, y, z, c=Y_test)
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
x = X_test.iloc[:, 0]
y = X_test.iloc[:, 2]
z = X_test.iloc[:, 3]
ax.scatter3D(x, y, z, c=Y_pred)
plt.show()


# Đánh giá và plot y_pred_random được tạo bằng cách random
Y_pred_random = np.random.randint(3,size=Y_test.shape)

print('Accuracy Random Predict:', metrics.accuracy_score(Y_test, Y_pred_random))

fig = plt.figure()
ax = plt.axes(projection='3d')
x = X_test.iloc[:, 0]
y = X_test.iloc[:, 2]
z = X_test.iloc[:, 3]
ax.scatter3D(x, y, z, c=Y_pred_random)
plt.show()