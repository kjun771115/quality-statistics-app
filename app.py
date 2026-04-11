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

    # 첫 번째 컬럼 사용
    col = df.columns[0]

    mean = df[col].mean()
    std = df[col].std()

    st.write("📈 평균:", mean)
    st.write("📉 표준편차:", std)

# Spec 입력
st.write("📏 Spec 입력")

lsl = st.number_input("LSL (하한)", value=10.0)
usl = st.number_input("USL (상한)", value=20.0)

# Cp / Cpk 계산
cp = (usl - lsl) / (6 * std)
cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

st.write("📊 Cp:", round(cp, 3))
st.write("📊 Cpk:", round(cpk, 3))

if cpk >= 1.33:
    st.success("공정 능력 양호 👍")
elif cpk >= 1.0:
    st.warning("공정 개선 필요 ⚠️")
else:
    st.error("공정 불량 위험 🚨")
import matplotlib.pyplot as plt

st.write("📊 분포 그래프")

fig, ax = plt.subplots()

# 히스토그램
ax.hist(df[col], bins=10)

# 평균선
ax.axvline(mean, linestyle='--', label='Mean')

# LSL / USL
ax.axvline(lsl, linestyle='--', label='LSL')
ax.axvline(usl, linestyle='--', label='USL')

ax.legend()

st.pyplot(fig)
