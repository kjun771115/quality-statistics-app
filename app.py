import streamlit as st
import pandas as pd

st.title("📊 도금 두께 분석 웹앱")

st.write("엑셀 파일을 업로드하세요")

file = st.file_uploader("파일 선택", type=["xlsx"])

if file:
    df = pd.read_excel(file)
    st.write("데이터 미리보기")
    st.dataframe(df)

    if "두께" in df.columns:
        mean = df["두께"].mean()
        std = df["두께"].std()

        st.write(f"평균: {mean:.2f}")
        st.write(f"표준편차: {std:.2f}")
