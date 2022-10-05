from statistics import mean
from turtle import color, position
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale

def one_hot(x):
    classes, index = np.unique(x, return_inverse= True)
    one_hot_vectors = np.zeros((x.shape[0], len(classes)))
    for i, cls in enumerate(index):
        one_hot_vectors[i, cls] = 1
    return one_hot_vectors

def l2(GT, PD):
    l2 = ((GT-PD)**2).sum()
    l2 = l2**0.5
    l2 /= len(GT)
    return l2

data = pd.read_csv('./Salary_Data.csv')
x = data.iloc[:, 0]
y = data.iloc[:, 1]
# plt.scatter(x, y)
# plt.show()

x=np.array(x).reshape(-1, 1)
y=np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=0.8, random_state=0)

model = LinearRegression()
model_scale = LinearRegression()
model_scale.fit(scale(X_train), Y_train)

Y_pred_scale = model_scale.predict(scale(X_test))
# print(Y_pred)
# print(Y_test)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print('MSE: ', mean_squared_error(Y_test, Y_pred))
print('L2: ', l2(Y_test, Y_pred))

print('MSE (scaled):', mean_squared_error(Y_test, Y_pred_scale))


print('L2 (scaled):', l2(Y_test, Y_pred_scale))

# plt.scatter(X_test, Y_test, color='Black')
# plt.scatter(X_test, Y_pred, color='Red')
# plt.show()


# ------------------------------------------

data_position = pd.read_csv('./Position_Salaries.csv')

x = data_position.iloc[:, :-1]
y = data_position.iloc[:, -1]

print(x)

# print(np.array(x[:, 0]))

x = np.array(x)
# print(x[:, 0])

position_one_hot = one_hot(x[:, 0])
# print(position_one_hot)

x_oh = np.concatenate((position_one_hot, x[:, 1:]), axis=-1)
print(x_oh)

X_train_oh, X_test_oh, Y_train, Y_test = train_test_split(x_oh, y, train_size = 0.8)
model_Pos = LinearRegression()
model_Pos.fit(X_train_oh, Y_train)

Y_pred = model_Pos.predict(X_test_oh)

print('MSE: ', mean_squared_error(Y_test, Y_pred))
print('L2: ', l2(Y_test, Y_pred))
# print(data)