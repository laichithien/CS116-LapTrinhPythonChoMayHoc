import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
import os
import seaborn as sb

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
st.header('Features selected: ')
feature_list = []
for i in range(len(list_header)):
    if feature_used[i]: # Features được dùng lưu trong feature_list
        st.write(list_header[i]) 
        feature_list.append(list_header[i])
# Lấy output
st.header("Output: ")
output_list = [header for header in list_header if header not in feature_list]
output = st.selectbox('Choose your output: ', output_list) # Output được lưu trong output 

st.header("Data Preprocessing")
train_test_split_col, k_folder_col = st.columns(2)

with train_test_split_col:
    train_size = st.slider('Train', 0, 100, format='%d%%') # Train size được lưu trong train_size
    
with k_folder_col:
    is_k_folder = st.checkbox("K Folder")
    if is_k_folder:
        k = st.number_input('Folder', 0, 10) # K folder (nếu có được lưu trong k)

