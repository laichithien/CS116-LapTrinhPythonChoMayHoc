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
st.title("HỆ THỐNG PHÂN TÍCH DỮ LIỆU TỰ ĐỘNG")

uploaded_file = st.file_uploader("Chọn ảnh đi bro")
if not uploaded_file: 
    st.warning("Please input a dataset")
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
train_test_split_col, k_folder_col = st.columns(2)

dataset_split = st.radio('Chọn cách chia dữ liệu huấn luyện', ['Train/test split', 'K Fold'])
if not dataset_split: 
    st.stop()

if dataset_split == 'Train/test split':
    train_size = st.slider('Train', 0, 100, format='%d%%') # Train size được lưu trong train_size
elif dataset_split == 'K Fold':
    k = st.slider('Folds', 0, 10) # K folder (nếu có được lưu trong k)



ohe = OneHotEncoder()
transformed = ohe.fit_transform(data_train_ob)
transformed = pd.DataFrame(transformed.astype(int).toarray())
st.table(transformed.iloc[0:10])

data_train = pd.concat([data_train_num, transformed], axis=1)

data_test =  dataset[output]
st.table(data_train.iloc[0: 10])
# Splitting strategy
if dataset_split == 'Train/test split':
    if not train_size:
        st.stop()
    X_train, X_test, y_train, y_test = train_test_split(data_train, data_test, train_size=train_size/100)

elif dataset_split == 'K Fold':
    kf = KFold(n_splits=k)
    kf.get_n_splits(data_train)

reg = LinearRegression()

reg.fit(X_train, y_train)

# st.write(y_test)
y_pred = reg.predict(X_test)
# st.write(y_pred)
mae = mean_absolute_error(y_test, y_pred)
st.write("MAE:", mae)
mse = mean_squared_error(y_test, y_pred)
st.write("MSE:", mse)
table1, table2 = st.columns(2)

with table1: 
    st.table(y_test)
with table2:
    st.table(y_pred)

