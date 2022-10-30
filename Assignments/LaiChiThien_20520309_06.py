
from unittest import result
import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import math
import seaborn as sns
import altair as alt 

st.title("HỆ THỐNG PHÂN TÍCH DỮ LIỆU TỰ ĐỘNG")


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
st.header("Đặc trưng đầu vào: ")
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
data_train_num_ob = pd.concat([data_train_num, data_train_ob], axis=1)
st.table(data_train_num_ob.iloc[0:10])


# st.table(data_train_ob.iloc[0:10])
# Lấy output
st.header("Đầu ra: ")
output_list = [header for header in list_header if header not in feature_list_root]
output = st.selectbox('Chọn đầu ra của mô hình dự đoán: ', output_list) # Output được lưu trong output 

st.header('Tương quan giữa output và các đặc trưng')
chosen_feature = [feature for feature in data_train_num_ob.columns.astype(str)]
tabs = st.tabs(chosen_feature)
for i in range(len(tabs)):
    with tabs[i]:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x=dataset[chosen_feature[i]], y=dataset[output])
        plt.xlabel(chosen_feature[i])
        plt.ylabel(output)
        st.pyplot(fig)

st.header('Tương quan giữa các đặc trưng')
feature1, feature2 = st.columns(2)

with feature1:
    ft1 = st.selectbox("Đặc trưng thứ nhất", feature_list_root)
    feature_selected = [feature for feature in feature_list_root if feature != ft1]
    st.write(ft1)
with feature2:
    ft2 = st.selectbox("Đặc trưng thứ hai", feature_selected)
    st.write(ft2)

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(x=dataset[ft1], y=dataset[ft2])
plt.xlabel(ft1)
plt.ylabel(ft2)
st.pyplot(fig)

st.header("Data Splitting")

ohe = OneHotEncoder()
transformed = ohe.fit_transform(data_train_ob)
transformed = pd.DataFrame(transformed.astype(int).toarray())
data_train = pd.concat([data_train_num, transformed], axis=1)
data_test =  dataset[output]


train_test_split_col, k_folder_col = st.columns(2)
with train_test_split_col:
    st.header("Train/test split")
    st.header("     ")
    train_size = st.slider('Train', 40, 100, format='%d%%') # Train size được lưu trong train_size
    
with k_folder_col:
    st.header("K Fold Cross-validation")
    k = st.slider('Folds', 2, 10)
    
run = st.button('Run')
if not run:
    st.stop()

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
    mae_tts = mean_absolute_error(y_test, y_pred)
    st.write("MAE:", mae_tts)
    mse_tts = mean_squared_error(y_test, y_pred)
    st.write("MSE:", mse_tts)
    r2_tts = r2_score(y_test, y_pred)
    st.write("R2:", r2_tts)
with result2:
    if not k:
        st.stop()
    kf = KFold(n_splits=k)
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
    mae_list = pd.DataFrame(mae_list, columns=['Score'])
    mse_list = pd.DataFrame(mse_list, columns=['Score'])
    r2_list = pd.DataFrame(r2_list, columns=['Score'])

    mae_list.index = mae_list.index.factorize()[0] + 1
    mse_list.index = mse_list.index.factorize()[0] + 1
    r2_list.index = r2_list.index.factorize()[0] + 1

    mae_list.reset_index(inplace=True)
    mae_list = mae_list.rename(columns = {'index':'Fold'})
    mse_list.reset_index(inplace=True)
    mse_list = mse_list.rename(columns = {'index':'Fold'})
    r2_list.reset_index(inplace=True)
    r2_list = r2_list.rename(columns = {'index':'Fold'})

    avg_mae = sum(mae_list['Score'])/len(mae_list)
    avg_mse = sum(mse_list['Score'])/len(mse_list)
    avg_r2 = sum(r2_list['Score'])/len(r2_list)
    st.write('Average MAE:', avg_mae)
    st.write('Average MSE:', avg_mse)
    st.write('Average R2:', avg_r2)
    graph_mae, graph_mse, graph_r2 = st.tabs(['MAE', 'MSE', 'R2'])
    
    with graph_mae:
        chart_mae = alt.Chart(mae_list).mark_bar(size=30).encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(chart_mae, use_container_width=False)
        # st.bar_chart(mae_list)
    with graph_mse:
        chart_mse = alt.Chart(mse_list).mark_bar().encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(chart_mae, use_container_width=True)
    with graph_r2:
        chart_r2 = alt.Chart(r2_list).mark_bar().encode(alt.X('Fold', axis=alt.Axis(title='Fold', tickMinStep=1)), y='Score')
        st.altair_chart(chart_mae, use_container_width=True)


    








