import streamlit as st
import hypothesis_app
import capability_app
import pandas as pd

st.title("품질 통계 분석 플랫폼")

menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["도금두께분석", "가설검정", "공정능력"]
)

if menu == "도금두께분석":
    st.header("📏 도금두께 분석")

    file = st.file_uploader("엑셀 파일 업로드", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.dataframe(df)

        col = df.columns[0]
        mean = df[col].mean()
        std = df[col].std()

        st.write(f"평균: {mean:.4f}")
        st.write(f"표준편차: {std:.4f}")

elif menu == "가설검정":
    hypothesis_app.run()

elif menu == "공정능력":
    capability_app.run()
