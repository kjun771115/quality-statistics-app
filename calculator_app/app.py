import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="간단한 계산기",
    page_icon="🧮",
    layout="centered"
)

# 제목 및 설명
st.title("🧮 간단한 계산기")
st.markdown("두 숫자를 입력하고 연산을 선택하세요.")
st.divider()

# 숫자 입력
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("첫 번째 숫자", value=0.0, format="%.4f", key="num1")
with col2:
    num2 = st.number_input("두 번째 숫자", value=0.0, format="%.4f", key="num2")

# 연산 선택
operation = st.radio(
    "연산 선택",
    options=["➕ 더하기", "➖ 빼기", "✖️ 곱하기", "➗ 나누기"],
    horizontal=True,
)

st.divider()

# 계산 실행
if st.button("계산하기", type="primary", use_container_width=True):
    op_map = {
        "➕ 더하기": "+",
        "➖ 빼기": "-",
        "✖️ 곱하기": "*",
        "➗ 나누기": "/",
    }
    op_symbol = op_map[operation]

    try:
        if op_symbol == "+":
            result = num1 + num2
        elif op_symbol == "-":
            result = num1 - num2
        elif op_symbol == "*":
            result = num1 * num2
        elif op_symbol == "/":
            if num2 == 0:
                raise ZeroDivisionError("0으로 나눌 수 없습니다.")
            result = num1 / num2

        # 결과를 정수로 딱 떨어지면 int로 표시
        display_result = int(result) if result == int(result) else round(result, 6)

        st.success(f"**{num1} {op_symbol} {num2} = {display_result}**")

    except ZeroDivisionError as e:
        st.error(f"⚠️ 오류: {e}")
    except Exception as e:
        st.error(f"⚠️ 예상치 못한 오류가 발생했습니다: {e}")

# 실행 안내
st.divider()
st.caption("💡 실행 방법: `streamlit run app.py`")
