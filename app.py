import streamlit as st
import pandas as pd

st.title("📊 도금 두께 분석 웹앱")

st.write("엑셀 파일을 업로드하세요")

file = st.file_uploader("파일 선택", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    st.success("업로드 완료!")

    st.write("📋 데이터 미리보기")
    st.dataframe(df)

    # 첫 번째 컬럼 자동 선택
    col = df.columns[0]

    mean = df[col].mean()
    std = df[col].std()

    st.write(f"📈 평균: {mean:.2f}")
    st.write(f"📉 표준편차: {std:.4f}")

      # Spec 입력
lsl = st.number_input("LSL (하한)", value=10.0)
usl = st.number_input("USL (상한)", value=20.0)

# Cp / Cpk 계산
cp = (usl - lsl) / (6 * std)
cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

st.write(f"📊 Cp: {cp:.3f}")
st.write(f"📊 Cpk: {cpk:.3f}")

# 판정
if cpk >= 1.33:
    st.success("✅ 공정 양호 (OK)")
else:
    st.error("🚨 공정 불량 위험 (NG)") 
