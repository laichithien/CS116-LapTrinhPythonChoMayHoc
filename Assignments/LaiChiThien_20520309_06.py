
from unittest import result
import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
import os
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
st.title("HỆ THỐNG PHÂN TÍCH DỮ LIỆU TỰ ĐỘNG")
import math

uploaded_file = st.file_uploader("Upload dữ liệu của bạn tại đây")
if not uploaded_file: 
    st.warning("Up đi ạ")
    st.stop()
# Nếu tồn tại file
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open(os.path.join('Assignments','data', uploaded_file.name), "wb") as f:
        f.write(bytes_data)
# Đọc file và visualize
dataset = pd.read_csv(uploaded_file)
st.table(dataset.iloc[0:10])
# Lấy feature
st.header("Input feature: ")
list_header = dataset.columns
checkboxes_feature = st.columns(len(list_header))
feature_used = []
for i in range(len(list_header)):
    with checkboxes_feature[i]:
        feature_used.append(st.checkbox(list_header[i]))
# st.header('Features selected: ')
feature_list_root = []
for i in range(len(list_header)):
    if feature_used[i]: # Features được dùng lưu trong feature_list_root
        # st.write(list_header[i]) 
        feature_list_root.append(list_header[i])

if not feature_list_root:
    st.stop()
feature_list = [feature for feature in feature_list_root if not dataset[feature].dtype == object]
nan_list = [header for header in dataset.columns if dataset[header].dtype == object and header in feature_list_root]
data_train_num = dataset.loc[:, feature_list]
data_train_ob = dataset.loc[:, nan_list]
st.table(pd.concat([data_train_num, data_train_ob], axis=1).iloc[0:10])
# st.table(data_train_ob.iloc[0:10])
# Lấy output
st.header("Output: ")
output_list = [header for header in list_header if header not in feature_list_root]
output = st.selectbox('Choose your output: ', output_list) # Output được lưu trong output 

st.header("Data Preprocessing")

ohe = OneHotEncoder()
transformed = ohe.fit_transform(data_train_ob)
transformed = pd.DataFrame(transformed.astype(int).toarray())
data_train = pd.concat([data_train_num, transformed], axis=1)
data_test =  dataset[output]


train_test_split_col, k_folder_col = st.columns(2)
with train_test_split_col:
    st.header("Train/test split")
    st.header("     ")
    train_size = st.slider('Train', 0, 100, format='%d%%') # Train size được lưu trong train_size
    
with k_folder_col:
    st.header("K Fold Cross-validation")
    k = st.slider('Folds', 0, 10)
    
result1, result2 = st.columns(2)
with result1:
    if not train_size:
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(data_train, data_test, train_size=train_size/100)
    reg = LinearRegression()
    if y_train.dtype == object:
        st.stop()
    reg.fit(X_train, y_train)
    # st.write(y_test)
    y_pred = reg.predict(X_test)
    # st.write(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write("MAE:", mae)
    mse = mean_squared_error(y_test, y_pred)
    st.write("MSE:", mse)
    r2 = r2_score(y_test, y_pred)
    st.write("R2:", r2)
with result2:
    if not k:
        st.stop()
    kf = KFold(n_splits=k)
    kf.get_n_splits(data_train)
    model_list = []
    mae_list = []
    mse_list = []
    r2_list = []
    for train_index, test_index in kf.split(data_train):
        X_train, X_test = data_train.values[train_index], data_train.values[test_index]
        y_train, y_test = data_test.values[train_index], data_test.values[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_list.append(model)
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)
    mae_list = pd.DataFrame(mae_list)
    mse_list = pd.DataFrame(mse_list)
    r2_list = pd.DataFrame(r2_list)
    avg_mae = sum(mae_list[0])/len(mae_list)
    avg_mse = sum(mse_list[0])/len(mse_list)
    avg_r2 = sum(r2_list[0])/len(r2_list)
    st.write('Average mae:', avg_mae)
    st.write('Average mse:', avg_mse)
    st.write('Average r2:', avg_r2)
# if dataset_split == 'Train/test split':
#     train_size = st.slider('Train', 0, 100, format='%d%%') # Train size được lưu trong train_size
# elif dataset_split == 'K Fold':
     # K folder (nếu có được lưu trong k)




# st.table(data_train.iloc[0: 10])
# Splitting strategy
# if dataset_split == 'Train/test split':
#     if not train_size:
#         st.stop()
#     X_train, X_test, y_train, y_test = train_test_split(data_train, data_test, train_size=train_size/100)


# elif dataset_split == 'K Fold':
    

# st.table(y_train)





