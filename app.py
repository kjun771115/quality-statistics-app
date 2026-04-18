import streamlit as st
import hypothesis_app
import capability_app

st.title("📊 품질 통계 분석 플랫폼")

menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["가설검정", "공정능력"]
)

if menu == "가설검정":
    hypothesis_app.run()

elif menu == "공정능력":
    capability_app.run()
