from zoneinfo import available_timezones
import streamlit as st 
import pandas as pd 
import random
import numpy as np
from PIL import Image

st.markdown(
    """
    # Đây là tutorial
    ## 1. Giới thiệu streamlit
    ## 2. Các thành phần cơ bản của giao diện
    """
)

a_value = st.text_input("Nhập a")
option = st.selectbox("Chọn phép toán", ["Cộng", "Trừ", "Nhân", "Chia"])
b_value = st.text_input("Nhập b")


# switch option

# if button:
    # st.text_input("Kết quả:", float(a_value) + float(b_value))

df = pd.DataFrame(
    np.random.randn(10,5),
    columns=('col %d' % i for i in range(5))
)
button = st.button("Tính")

if button:
    if option == "Cộng":
        st.text_input("Kết quả", float(a_value) + float(b_value))
    elif option == "Trừ":
        st.text_input("Kết quả", float(a_value) - float(b_value))
    if option == "Nhân":
        st.text_input("Kết quả", float(a_value) * float(b_value))
    elif option == "Chia":
        st.text_input("Kết quả", float(a_value) / float(b_value))

st.table(df)
st.line_chart(df)
girl = Image.open("C:/Workspace/CS116-LapTrinhPythonChoMayHoc/Assignments/girl.jpg")

col1, col2, col3 = st.columns(3)
with col1:
    st.header("Col1")
    st.image(girl)
with col2:
    st.header("Col2")
    st.image(girl)
with col3:
    st.header("Col3")
    st.image(girl)


tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])


with tab1: 
    st.header("Blah")
    st.image(girl)
with tab2: 
    st.header("Blah")
    st.image(girl)
with tab3: 
    st.header("Blah")
    st.image(girl)

uploaded_file = st.file_uploader("Chọn ảnh đi bro")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open('Assignments/data/'+uploaded_file.name, "wb") as f:
        f.write(bytes_data)

pic = st.image("Assignments/data/"+uploaded_file.name)