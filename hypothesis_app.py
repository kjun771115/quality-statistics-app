# =============================================================================
# 가설검정 전문 분석 웹앱 (Hypothesis Testing Professional Analyzer)
# =============================================================================
# 대상: 품질관리기술사 / 통계 전문가 / 제조업 품질 담당자
# 기능: 7종 가설검정, 시각화, 전문가 수준 해석 제공
# 구조: 각 검정 함수 분리, SPC/Cpk 연계 확장 가능 구조
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import warnings
import io
import os

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 전역 설정 / 한글 폰트 설정
# =============================================================================

def setup_korean_font():
    """matplotlib 한글 폰트 설정 (NanumGothic 또는 시스템 폰트 자동 탐색)"""
    font_candidates = [
        "NanumGothic", "NanumBarunGothic", "Malgun Gothic",
        "AppleGothic", "DejaVu Sans"
    ]
    available = [f.name for f in font_manager.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            matplotlib.rc("font", family=font)
            break
    else:
        # 폴백: 유니코드 마이너스 기호 처리만
        pass
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_korean_font()

# =============================================================================
# 2. 페이지 기본 설정
# =============================================================================

st.set_page_config(
    page_title="가설검정 전문 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 (전문가용 다크 테마 강조)
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1E3A5F;
        border-bottom: 3px solid #2E86C1; padding-bottom: 8px; margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #2E86C1;
        background: #EBF5FB; padding: 8px 14px; border-radius: 6px;
        border-left: 4px solid #2E86C1; margin: 15px 0 10px 0;
    }
    .result-box {
        background: #F8F9FA; border: 1px solid #DEE2E6;
        border-radius: 8px; padding: 16px; margin: 10px 0;
    }
    .reject-box {
        background: #FDEDEC; border: 2px solid #E74C3C;
        border-radius: 8px; padding: 16px; margin: 10px 0;
    }
    .accept-box {
        background: #EAFAF1; border: 2px solid #27AE60;
        border-radius: 8px; padding: 16px; margin: 10px 0;
    }
    .metric-card {
        background: #2E86C1; color: white; border-radius: 8px;
        padding: 12px 18px; text-align: center; margin: 5px;
    }
    .interpret-box {
        background: #FEF9E7; border-left: 4px solid #F39C12;
        padding: 14px; border-radius: 0 8px 8px 0; margin: 10px 0;
        font-size: 0.95rem; line-height: 1.7;
    }
    .warning-box {
        background: #FDF2F8; border-left: 4px solid #8E44AD;
        padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 8px 0;
    }
    hr.divider { border: none; border-top: 2px solid #2E86C1; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. 데이터 로드 유틸리티
# =============================================================================

def load_data(uploaded_file) -> pd.DataFrame:
    """CSV 또는 Excel 파일을 DataFrame으로 로드"""
    try:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "csv":
            # 인코딩 자동 탐지 (UTF-8, EUC-KR 순서 시도)
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="euc-kr")
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드하세요.")
            return None
        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return None


def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """선택된 컬럼을 수치형으로 변환하고 NaN 제거"""
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return series


def validate_sample(data: np.ndarray, min_n: int = 3, label: str = "데이터") -> bool:
    """샘플 크기 및 분산 유효성 검사"""
    if len(data) < min_n:
        st.error(f"⚠️ {label}: 유효 데이터가 {len(data)}개로 최소 {min_n}개 이상 필요합니다.")
        return False
    if np.std(data) == 0:
        st.warning(f"⚠️ {label}: 모든 값이 동일하여 분산이 0입니다. 검정 결과가 부정확할 수 있습니다.")
    return True


# =============================================================================
# 4. 검정 함수 모듈 (각 검정별 독립 함수)
# =============================================================================

# --- 4-1. 1-Sample t-test ---------------------------------------------------

def run_one_sample_ttest(data: np.ndarray, mu0: float, alpha: float,
                          alternative: str) -> dict:
    """
    1-표본 t-검정
    H0: μ = μ0  vs  H1: μ ≠ μ0 (양측) 또는 단측

    Parameters
    ----------
    data        : 측정값 배열
    mu0         : 귀무가설 목표값
    alpha       : 유의수준
    alternative : 'two-sided' | 'greater' | 'less'

    Returns
    -------
    dict: 검정 통계량, p-value, 자유도, 신뢰구간 등
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    df = n - 1

    t_stat, p_value = stats.ttest_1samp(data, popmean=mu0,
                                         alternative=alternative)

    # 신뢰구간 (양측 기준)
    ci_level = 1 - alpha
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se

    return {
        "test_name": "1-표본 t-검정 (1-Sample t-test)",
        "n": n, "mean": mean, "std": std, "se": se,
        "t_stat": t_stat, "df": df, "p_value": p_value,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_level": ci_level,
        "alpha": alpha, "alternative": alternative, "mu0": mu0,
        "reject": p_value < alpha
    }


# --- 4-2. 2-Sample t-test ---------------------------------------------------

def run_two_sample_ttest(data1: np.ndarray, data2: np.ndarray,
                          alpha: float, alternative: str,
                          equal_var: bool = False) -> dict:
    """
    독립 2-표본 t-검정 (Welch's t-test 기본)
    H0: μ1 = μ2  vs  H1: μ1 ≠ μ2 (또는 단측)
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)

    t_stat, p_value = stats.ttest_ind(data1, data2,
                                       equal_var=equal_var,
                                       alternative=alternative)

    # Welch df 근사
    if not equal_var:
        se1_sq = std1**2 / n1
        se2_sq = std2**2 / n2
        df = (se1_sq + se2_sq)**2 / (
            se1_sq**2 / (n1 - 1) + se2_sq**2 / (n2 - 1)
        )
    else:
        df = n1 + n2 - 2

    # 평균 차이 신뢰구간
    diff = mean1 - mean2
    se_diff = np.sqrt(std1**2/n1 + std2**2/n2)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    return {
        "test_name": "2-표본 t-검정 (2-Sample t-test, Welch)",
        "n1": n1, "n2": n2,
        "mean1": mean1, "mean2": mean2,
        "std1": std1, "std2": std2,
        "t_stat": t_stat, "df": df, "p_value": p_value,
        "diff": diff,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_level": 1 - alpha,
        "alpha": alpha, "alternative": alternative,
        "equal_var": equal_var, "reject": p_value < alpha
    }


# --- 4-3. Paired t-test -----------------------------------------------------

def run_paired_ttest(data1: np.ndarray, data2: np.ndarray,
                      alpha: float, alternative: str) -> dict:
    """
    대응표본 t-검정
    H0: μD = 0  vs  H1: μD ≠ 0 (또는 단측)
    """
    if len(data1) != len(data2):
        return {"error": "대응표본 t-검정은 두 그룹의 데이터 수가 동일해야 합니다."}

    diff = data1 - data2
    n = len(diff)
    mean_d = np.mean(diff)
    std_d = np.std(diff, ddof=1)
    se_d = std_d / np.sqrt(n)
    df = n - 1

    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d

    return {
        "test_name": "대응표본 t-검정 (Paired t-test)",
        "n": n, "mean_d": mean_d, "std_d": std_d, "se_d": se_d,
        "t_stat": t_stat, "df": df, "p_value": p_value,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_level": 1 - alpha,
        "alpha": alpha, "alternative": alternative,
        "diff_data": diff, "reject": p_value < alpha
    }


# --- 4-4. Proportion test ---------------------------------------------------

def run_proportion_test(x: int, n: int, p0: float,
                         alpha: float, alternative: str) -> dict:
    """
    불량률 검정 (Z-검정 기반, 정규근사)
    H0: p = p0  vs  H1: p ≠ p0 (또는 단측)
    """
    p_hat = x / n
    se = np.sqrt(p0 * (1 - p0) / n)

    if se == 0:
        return {"error": "분모가 0이 됩니다. p0 및 n 값을 확인하세요."}

    z_stat = (p_hat - p0) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        ci_lower = p_hat - stats.norm.ppf(1 - alpha / 2) * np.sqrt(p_hat * (1 - p_hat) / n)
        ci_upper = p_hat + stats.norm.ppf(1 - alpha / 2) * np.sqrt(p_hat * (1 - p_hat) / n)
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
        ci_lower = p_hat - stats.norm.ppf(1 - alpha) * np.sqrt(p_hat * (1 - p_hat) / n)
        ci_upper = 1.0
    else:  # less
        p_value = stats.norm.cdf(z_stat)
        ci_lower = 0.0
        ci_upper = p_hat + stats.norm.ppf(1 - alpha) * np.sqrt(p_hat * (1 - p_hat) / n)

    ci_lower = max(0.0, ci_lower)
    ci_upper = min(1.0, ci_upper)

    return {
        "test_name": "불량률 검정 (Proportion Test, Z-검정)",
        "x": x, "n": n, "p_hat": p_hat, "p0": p0,
        "z_stat": z_stat, "p_value": p_value,
        "ci_lower": ci_lower, "ci_upper": ci_upper, "ci_level": 1 - alpha,
        "alpha": alpha, "alternative": alternative,
        "reject": p_value < alpha
    }


# --- 4-5. Poisson test ------------------------------------------------------

def run_poisson_test(c_obs: int, c0: float, n_units: int,
                      alpha: float, alternative: str) -> dict:
    """
    불량수 검정 (Poisson 검정)
    H0: λ = c0 (단위당 불량수)  vs  H1: λ ≠ c0 (또는 단측)

    scipy의 정확 Poisson 검정 사용 (chi2 분포 기반)
    """
    # 기대 불량수
    expected = c0 * n_units

    # 정확 Poisson 검정 (chi-square 근사)
    if alternative == "two-sided":
        # 양쪽 꼬리 합산
        p_lower = stats.poisson.cdf(c_obs, expected)
        p_upper = 1 - stats.poisson.cdf(c_obs - 1, expected)
        p_value = 2 * min(p_lower, p_upper)
        p_value = min(p_value, 1.0)
    elif alternative == "greater":
        p_value = 1 - stats.poisson.cdf(c_obs - 1, expected)
    else:
        p_value = stats.poisson.cdf(c_obs, expected)

    # 신뢰구간 (Garwood 방법, chi2 기반)
    if c_obs == 0:
        ci_lower_count = 0.0
    else:
        ci_lower_count = stats.chi2.ppf(alpha / 2, 2 * c_obs) / 2
    ci_upper_count = stats.chi2.ppf(1 - alpha / 2, 2 * (c_obs + 1)) / 2

    lambda_hat = c_obs / n_units

    return {
        "test_name": "불량수 검정 (Poisson Test)",
        "c_obs": c_obs, "n_units": n_units,
        "lambda_hat": lambda_hat, "c0": c0, "expected": expected,
        "p_value": p_value,
        "ci_lower_count": ci_lower_count,
        "ci_upper_count": ci_upper_count,
        "ci_lower_lambda": ci_lower_count / n_units,
        "ci_upper_lambda": ci_upper_count / n_units,
        "ci_level": 1 - alpha,
        "alpha": alpha, "alternative": alternative,
        "reject": p_value < alpha
    }


# --- 4-6. 정규성 검정 (Shapiro-Wilk) ----------------------------------------

def run_shapiro_wilk(data: np.ndarray, alpha: float) -> dict:
    """
    Shapiro-Wilk 정규성 검정
    H0: 데이터가 정규분포를 따름
    H1: 데이터가 정규분포를 따르지 않음

    주의: n ≤ 5000 에서 신뢰도 높음
    """
    n = len(data)
    if n > 5000:
        st.warning("⚠️ 샘플 크기가 5,000을 초과합니다. Shapiro-Wilk 검정의 신뢰도가 낮아질 수 있습니다.")

    w_stat, p_value = stats.shapiro(data)

    # 추가 기술통계
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)

    return {
        "test_name": "정규성 검정 (Shapiro-Wilk Test)",
        "n": n, "w_stat": w_stat, "p_value": p_value,
        "skew": skew, "kurtosis": kurt,
        "mean": np.mean(data), "std": np.std(data, ddof=1),
        "alpha": alpha,
        "reject": p_value < alpha   # True이면 정규분포 기각
    }


# --- 4-7. 등분산 검정 (Levene) -----------------------------------------------

def run_levene_test(*groups: np.ndarray, alpha: float,
                    center: str = "median") -> dict:
    """
    Levene 등분산 검정
    H0: 모든 그룹의 분산이 동일 (σ1² = σ2² = ...)
    H1: 적어도 하나의 그룹 분산이 다름

    center='median': Brown-Forsythe 검정 (이상치에 강건)
    center='mean'  : 고전적 Levene
    """
    valid_groups = [g for g in groups if len(g) >= 2]
    if len(valid_groups) < 2:
        return {"error": "등분산 검정에는 2개 이상의 유효한 그룹이 필요합니다."}

    f_stat, p_value = stats.levene(*valid_groups, center=center)
    df1 = len(valid_groups) - 1
    df2 = sum(len(g) for g in valid_groups) - len(valid_groups)

    group_stats = [
        {"n": len(g), "mean": np.mean(g), "std": np.std(g, ddof=1),
         "var": np.var(g, ddof=1)}
        for g in valid_groups
    ]

    return {
        "test_name": f"등분산 검정 (Levene Test, center={center})",
        "n_groups": len(valid_groups),
        "f_stat": f_stat, "df1": df1, "df2": df2,
        "p_value": p_value,
        "group_stats": group_stats,
        "alpha": alpha, "center": center,
        "reject": p_value < alpha
    }


# =============================================================================
# 5. 시각화 함수
# =============================================================================

def plot_histogram_with_normal(data: np.ndarray, title: str,
                                 mean: float = None, std: float = None,
                                 mu0: float = None) -> plt.Figure:
    """히스토그램 + 정규분포 곡선 시각화"""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    mean = mean or np.mean(data)
    std = std or np.std(data, ddof=1)

    # 히스토그램
    n_bins = min(int(np.sqrt(len(data))) + 2, 30)
    ax.hist(data, bins=n_bins, density=True, color="#3498DB", alpha=0.65,
            edgecolor="white", label="관측 데이터")

    # 정규분포 곡선
    x_range = np.linspace(mean - 4 * std, mean + 4 * std, 300)
    ax.plot(x_range, stats.norm.pdf(x_range, mean, std),
            color="#E74C3C", lw=2.5, label=f"정규분포 N({mean:.3f}, {std:.3f}²)")

    # 목표값 표시
    if mu0 is not None:
        ax.axvline(mu0, color="#F39C12", lw=2, linestyle="--",
                   label=f"목표값 μ₀ = {mu0}")

    # 표본평균 표시
    ax.axvline(mean, color="#2ECC71", lw=2, linestyle="-.",
               label=f"표본평균 x̄ = {mean:.4f}")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("측정값", fontsize=11)
    ax.set_ylabel("밀도 (Density)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_boxplot_comparison(data1: np.ndarray, data2: np.ndarray,
                              label1: str = "그룹 1", label2: str = "그룹 2",
                              title: str = "박스플롯 비교") -> plt.Figure:
    """두 그룹 Boxplot 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = ["#3498DB", "#E74C3C"]
    groups = [data1, data2]
    labels = [label1, label2]

    # 좌: Boxplot
    bp = axes[0].boxplot(groups, labels=labels, patch_artist=True,
                          notch=False, widths=0.45)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2.5)

    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("측정값", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # 우: 히스토그램 겹치기
    for g, lbl, c in zip(groups, labels, colors):
        n_bins = min(int(np.sqrt(len(g))) + 2, 25)
        axes[1].hist(g, bins=n_bins, density=True, color=c, alpha=0.55,
                     edgecolor="white", label=lbl)
        m, s = np.mean(g), np.std(g, ddof=1)
        x_r = np.linspace(m - 4 * s, m + 4 * s, 200)
        axes[1].plot(x_r, stats.norm.pdf(x_r, m, s), color=c, lw=2)

    axes[1].set_title("분포 비교", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("측정값", fontsize=10)
    axes[1].set_ylabel("밀도", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_t_rejection_region(t_stat: float, df: float, alpha: float,
                              alternative: str, title: str = "검정 기각역") -> plt.Figure:
    """t-분포 기각역 시각화"""
    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.linspace(-5, 5, 500)
    y = stats.t.pdf(x, df)
    ax.plot(x, y, color="#2C3E50", lw=2.5, label=f"t-분포 (df={df:.1f})")

    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        ax.fill_between(x, y, where=(x <= -t_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (각 α/2={alpha/2:.3f})")
        ax.fill_between(x, y, where=(x >= t_crit), color="#E74C3C", alpha=0.4)
        ax.axvline(-t_crit, color="#C0392B", lw=1.5, linestyle="--")
        ax.axvline(t_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 ±{t_crit:.3f}")
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df)
        ax.fill_between(x, y, where=(x >= t_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (α={alpha})")
        ax.axvline(t_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 {t_crit:.3f}")
    else:
        t_crit = stats.t.ppf(alpha, df)
        ax.fill_between(x, y, where=(x <= t_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (α={alpha})")
        ax.axvline(t_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 {t_crit:.3f}")

    # 검정통계량 표시
    ax.axvline(t_stat, color="#27AE60", lw=2.5, linestyle="-",
               label=f"검정통계량 t = {t_stat:.4f}")
    ax.scatter([t_stat], [stats.t.pdf(t_stat, df)], color="#27AE60", s=80, zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("t-값", fontsize=10)
    ax.set_ylabel("확률밀도", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_z_rejection_region(z_stat: float, alpha: float,
                              alternative: str, title: str = "검정 기각역") -> plt.Figure:
    """Z-분포 기각역 시각화"""
    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.linspace(-4.5, 4.5, 500)
    y = stats.norm.pdf(x)
    ax.plot(x, y, color="#2C3E50", lw=2.5, label="표준정규분포 N(0,1)")

    if alternative == "two-sided":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(x, y, where=(x <= -z_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (각 α/2={alpha/2:.3f})")
        ax.fill_between(x, y, where=(x >= z_crit), color="#E74C3C", alpha=0.4)
        ax.axvline(-z_crit, color="#C0392B", lw=1.5, linestyle="--")
        ax.axvline(z_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 ±{z_crit:.3f}")
    elif alternative == "greater":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(x, y, where=(x >= z_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (α={alpha})")
        ax.axvline(z_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 {z_crit:.3f}")
    else:
        z_crit = stats.norm.ppf(alpha)
        ax.fill_between(x, y, where=(x <= z_crit), color="#E74C3C", alpha=0.4,
                        label=f"기각역 (α={alpha})")
        ax.axvline(z_crit, color="#C0392B", lw=1.5, linestyle="--",
                   label=f"임계값 {z_crit:.3f}")

    ax.axvline(z_stat, color="#27AE60", lw=2.5,
               label=f"검정통계량 z = {z_stat:.4f}")
    ax.scatter([z_stat], [stats.norm.pdf(z_stat)], color="#27AE60", s=80, zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("z-값", fontsize=10)
    ax.set_ylabel("확률밀도", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_qq_plot(data: np.ndarray, title: str = "Q-Q Plot (정규성 확인)") -> plt.Figure:
    """정규성 Q-Q Plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax.scatter(osm, osr, color="#3498DB", alpha=0.7, s=40, label="관측값")
    x_line = np.array([min(osm), max(osm)])
    ax.plot(x_line, slope * x_line + intercept, color="#E74C3C", lw=2,
            label=f"기준선 (R²={r**2:.4f})")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("이론적 분위수 (Theoretical Quantiles)", fontsize=10)
    ax.set_ylabel("표본 분위수 (Sample Quantiles)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_variance_comparison(groups: list, labels: list,
                              title: str = "분산 비교") -> plt.Figure:
    """그룹별 분산 비교 (Levene 검정 보조)"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Boxplot
    bp = axes[0].boxplot(groups, labels=labels, patch_artist=True, widths=0.5)
    colors_list = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
                   "#9B59B6", "#1ABC9C"]
    for i, (patch, lbl) in enumerate(zip(bp["boxes"], labels)):
        patch.set_facecolor(colors_list[i % len(colors_list)])
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("white")
        med.set_linewidth(2)
    axes[0].set_title("그룹별 분포 비교 (Boxplot)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("측정값", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # 표준편차 막대그래프
    stds = [np.std(g, ddof=1) for g in groups]
    bars = axes[1].bar(labels, stds,
                        color=[colors_list[i % len(colors_list)]
                               for i in range(len(labels))],
                        alpha=0.75, edgecolor="white")
    for bar, val in zip(bars, stds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("그룹별 표준편차", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("표준편차 (σ)", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# 6. 결과 출력 함수 (공통 레이아웃)
# =============================================================================

ALT_KOR = {
    "two-sided": "양측 (≠)",
    "greater": "단측 우측 (>)",
    "less": "단측 좌측 (<)"
}


def display_result_header(result: dict):
    """검정 결과 헤더 및 핵심 지표 출력"""
    reject = result.get("reject", False)

    # 기각 여부 강조 박스
    if reject:
        st.markdown(
            f'<div class="reject-box"><b>🔴 귀무가설 기각 (Reject H₀)</b><br>'
            f'p-value = <b>{result["p_value"]:.6f}</b> &lt; α = {result["alpha"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="accept-box"><b>🟢 귀무가설 채택 (Fail to Reject H₀)</b><br>'
            f'p-value = <b>{result["p_value"]:.6f}</b> ≥ α = {result["alpha"]}</div>',
            unsafe_allow_html=True
        )


def display_metric_row(metrics: list):
    """
    핵심 통계량을 열 형태로 표시
    metrics: [(label, value), ...]
    """
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)


def display_interpretation(result: dict, test_type: str,
                             context: dict = None):
    """
    품질 관점 실무 해석문 생성 및 출력

    Parameters
    ----------
    result    : 검정 결과 딕셔너리
    test_type : 검정 유형 식별자
    context   : 추가 맥락 정보 (컬럼명, 공정명 등)
    """
    p = result["p_value"]
    alpha = result["alpha"]
    reject = result["reject"]
    alt_kor = ALT_KOR.get(result.get("alternative", "two-sided"), "양측")

    interp = ""

    if test_type == "1sample_t":
        mu0 = result["mu0"]
        mean = result["mean"]
        ci_l, ci_u = result["ci_lower"], result["ci_upper"]
        if reject:
            interp = (
                f"p-value({p:.4f})가 유의수준 α={alpha}보다 **작으므로** 귀무가설을 기각합니다.\n\n"
                f"이는 공정 평균({mean:.4f})이 목표값 μ₀={mu0}과 통계적으로 **유의미한 차이**가 있음을 의미합니다.\n\n"
                f"**품질 관점 해석:** 공정이 목표값에서 벗어나 있습니다. "
                f"치우침(Bias) 원인을 분석하고 공정 조정 또는 원인 조치가 필요합니다.\n\n"
                f"**{int((1-alpha)*100)}% 신뢰구간:** [{ci_l:.4f}, {ci_u:.4f}] — 이 구간에 μ₀={mu0}이 포함되지 않음을 확인하세요."
            )
        else:
            interp = (
                f"p-value({p:.4f})가 유의수준 α={alpha}보다 **크므로** 귀무가설을 기각하지 못합니다.\n\n"
                f"현재 데이터로는 공정 평균이 목표값 μ₀={mu0}과 다르다는 증거가 충분하지 않습니다.\n\n"
                f"**품질 관점 해석:** 공정이 목표값 수준에서 안정적으로 운영되고 있습니다. "
                f"그러나 샘플 크기(n={result['n']})가 충분한지 검정력(Power)도 확인하세요.\n\n"
                f"**{int((1-alpha)*100)}% 신뢰구간:** [{ci_l:.4f}, {ci_u:.4f}]"
            )

    elif test_type == "2sample_t":
        diff = result["diff"]
        if reject:
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설을 기각합니다.\n\n"
                f"두 그룹의 평균(Δ={diff:.4f})에는 통계적으로 **유의미한 차이**가 있습니다.\n\n"
                f"**품질 관점 해석:** 두 공정/설비/작업자 간 차이가 존재합니다. "
                f"MSA 또는 공정 비교 분석을 통해 변동 원인을 규명하세요.\n\n"
                f"**평균차이 {int((1-alpha)*100)}% CI:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설을 기각하지 못합니다.\n\n"
                f"두 그룹의 평균 차이(Δ={diff:.4f})는 통계적으로 **유의미하지 않습니다**.\n\n"
                f"**품질 관점 해석:** 두 그룹 간 공정 수준이 동등하다고 볼 수 있습니다. "
                f"설비/작업자 교체에 따른 품질 영향이 없음을 시사합니다.\n\n"
                f"**평균차이 {int((1-alpha)*100)}% CI:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            )

    elif test_type == "paired_t":
        mean_d = result["mean_d"]
        if reject:
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설을 기각합니다.\n\n"
                f"개선 전·후 평균 차이(D̄={mean_d:.4f})가 통계적으로 **유의미**합니다.\n\n"
                f"**품질 관점 해석:** 공정 개선 조치(5M1E 변경, 설비 교체 등)가 품질 특성에 실질적인 영향을 미쳤습니다. "
                f"개선 효과를 공식 인정하고 표준화를 진행하세요.\n\n"
                f"**차이 {int((1-alpha)*100)}% CI:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설을 기각하지 못합니다.\n\n"
                f"개선 전·후 평균 차이(D̄={mean_d:.4f})가 통계적으로 **유의미하지 않습니다**.\n\n"
                f"**품질 관점 해석:** 현재 개선 조치의 효과가 데이터로 입증되지 않았습니다. "
                f"추가 샘플 수집 또는 개선 방법 재검토가 필요합니다."
            )

    elif test_type == "proportion":
        p_hat = result["p_hat"]
        p0 = result["p0"]
        if reject:
            direction = "높아" if p_hat > p0 else "낮아"
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설을 기각합니다.\n\n"
                f"관측 불량률({p_hat:.4%})이 기준 불량률 p₀={p0:.4%}보다 통계적으로 **{direction}졌습니다**.\n\n"
                f"**품질 관점 해석:** 공정 불량률이 기준과 다릅니다. "
                f"불량 원인 분석(4M 변동, 설계 마진 등)을 즉시 시행하고 고객 납품 일정을 점검하세요.\n\n"
                f"**불량률 {int((1-alpha)*100)}% CI:** [{result['ci_lower']:.4%}, {result['ci_upper']:.4%}]"
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설을 기각하지 못합니다.\n\n"
                f"현재 불량률({p_hat:.4%})이 기준 불량률 p₀={p0:.4%}과 통계적으로 **차이가 없습니다**.\n\n"
                f"**품질 관점 해석:** 공정 불량률이 관리 기준 내에 있습니다. "
                f"지속적인 모니터링(P-Chart 운영)을 통해 안정성을 유지하세요.\n\n"
                f"**불량률 {int((1-alpha)*100)}% CI:** [{result['ci_lower']:.4%}, {result['ci_upper']:.4%}]"
            )

    elif test_type == "poisson":
        lambda_hat = result["lambda_hat"]
        c0 = result["c0"]
        if reject:
            direction = "많아" if lambda_hat > c0 else "적어"
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설을 기각합니다.\n\n"
                f"단위당 불량수({lambda_hat:.4f})가 기준값 c₀={c0}보다 통계적으로 **{direction}졌습니다**.\n\n"
                f"**품질 관점 해석:** 공정 결함 발생률이 기준과 다릅니다. "
                f"C-Chart 또는 U-Chart를 통한 관리도 분석과 특수원인 조사를 진행하세요."
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설을 기각하지 못합니다.\n\n"
                f"단위당 불량수({lambda_hat:.4f})가 기준값 c₀={c0}과 통계적으로 **차이가 없습니다**.\n\n"
                f"**품질 관점 해석:** 결함 발생률이 관리 기준 내에서 안정적입니다. "
                f"C-Chart를 계속 운영하며 추세 변화를 모니터링하세요."
            )

    elif test_type == "shapiro":
        if reject:
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설(정규분포)을 기각합니다.\n\n"
                f"**데이터가 정규분포를 따르지 않을 가능성이 높습니다.**\n\n"
                f"왜도(Skewness): {result['skew']:.4f}, 첨도(Kurtosis): {result['kurtosis']:.4f}\n\n"
                f"**품질 관점 해석:** 공정 데이터의 비정규성 원인을 분석하세요 "
                f"(혼합분포, 이상치, 규격 한계 절단 등). "
                f"SPC나 공정능력 분석 시 비정규 방법론(Box-Cox 변환, 비모수 검정) 적용을 검토하세요."
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설(정규분포)을 기각하지 못합니다.\n\n"
                f"**현재 데이터로는 정규분포 가정을 기각할 수 없습니다.**\n\n"
                f"왜도(Skewness): {result['skew']:.4f}, 첨도(Kurtosis): {result['kurtosis']:.4f}\n\n"
                f"**품질 관점 해석:** t-검정, ANOVA, Cpk 등 정규성 가정 기반 분석을 사용할 수 있습니다. "
                f"Q-Q Plot으로 꼬리 부분의 이탈도 함께 확인하세요."
            )

    elif test_type == "levene":
        if reject:
            interp = (
                f"p-value({p:.4f}) < α={alpha}이므로 귀무가설(등분산)을 기각합니다.\n\n"
                f"**그룹 간 분산이 통계적으로 유의미하게 다릅니다 (이분산).**\n\n"
                f"**품질 관점 해석:** 설비/작업자/시간대별 산포가 다릅니다. "
                f"이분산 원인을 규명하고, 2-표본 t-검정 시 Welch 검정을 사용하세요. "
                f"공정능력 분석 시 그룹별로 분리 분석하는 것을 권장합니다."
            )
        else:
            interp = (
                f"p-value({p:.4f}) ≥ α={alpha}이므로 귀무가설(등분산)을 기각하지 못합니다.\n\n"
                f"**그룹 간 분산이 통계적으로 동일하다고 볼 수 있습니다 (등분산).**\n\n"
                f"**품질 관점 해석:** 설비/작업자/시간대별 산포가 유사하여 "
                f"공정이 안정적임을 나타냅니다. 등분산 가정이 필요한 t-검정이나 ANOVA를 안심하고 적용할 수 있습니다."
            )

    if interp:
        st.markdown('<div class="interpret-box">📋 <b>실무 해석</b><br><br>' +
                    interp.replace("\n", "<br>") + '</div>',
                    unsafe_allow_html=True)


# =============================================================================
# 7. 사이드바 설정 UI
# =============================================================================

def render_sidebar() -> dict:
    """사이드바 분석 설정 UI, 설정값 딕셔너리 반환"""
    st.sidebar.markdown("## ⚙️ 분석 설정")
    st.sidebar.markdown("---")

    # 검정 선택
    test_options = {
        "① 1-표본 t-검정 (1-Sample t-test)": "1sample_t",
        "② 2-표본 t-검정 (2-Sample t-test)": "2sample_t",
        "③ 대응표본 t-검정 (Paired t-test)": "paired_t",
        "④ 불량률 검정 (Proportion Test)": "proportion",
        "⑤ 불량수 검정 (Poisson Test)": "poisson",
        "⑥ 정규성 검정 (Shapiro-Wilk)": "shapiro",
        "⑦ 등분산 검정 (Levene Test)": "levene",
    }
    selected_label = st.sidebar.selectbox(
        "🔬 검정 방법 선택", list(test_options.keys())
    )
    test_type = test_options[selected_label]

    st.sidebar.markdown("---")

    # 유의수준
    alpha = st.sidebar.number_input(
        "유의수준 α (alpha)", min_value=0.001, max_value=0.20,
        value=0.05, step=0.01, format="%.3f",
        help="일반적으로 0.05 또는 0.01 사용"
    )

    # 가설 방향 (proportion/poisson은 해당 없음으로 단순화)
    alt_map = {
        "양측 검정 (H₁: ≠)": "two-sided",
        "단측 검정 우측 (H₁: >)": "greater",
        "단측 검정 좌측 (H₁: <)": "less"
    }
    if test_type in ["shapiro", "levene"]:
        alternative = "two-sided"
    else:
        alt_label = st.sidebar.selectbox(
            "📐 대립가설 방향",
            list(alt_map.keys()),
            help="양측: 차이 존재 여부 / 단측: 방향까지 검정"
        )
        alternative = alt_map[alt_label]

    st.sidebar.markdown("---")

    # 검정별 추가 파라미터
    params = {}

    if test_type == "1sample_t":
        params["mu0"] = st.sidebar.number_input(
            "목표값 μ₀ (귀무가설 기준값)",
            value=0.0, step=0.1, format="%.4f",
            help="예: 공정 목표 두께 18.0 μm"
        )

    elif test_type == "2sample_t":
        params["equal_var"] = st.sidebar.checkbox(
            "등분산 가정 (Pooled t-test)",
            value=False,
            help="체크 해제 시 Welch's t-test (이분산 가정) 사용"
        )

    elif test_type == "proportion":
        params["p0"] = st.sidebar.number_input(
            "기준 불량률 p₀",
            min_value=0.0001, max_value=0.9999,
            value=0.05, step=0.001, format="%.4f",
            help="예: 목표 불량률 5% → 0.05"
        )
        params["x_input"] = st.sidebar.number_input(
            "불량 개수 x", min_value=0, value=5, step=1,
            help="검사한 샘플 중 불량품 수"
        )
        params["n_input"] = st.sidebar.number_input(
            "전체 샘플 수 n", min_value=1, value=100, step=1
        )

    elif test_type == "poisson":
        params["c0"] = st.sidebar.number_input(
            "기준 불량수(단위당) c₀",
            min_value=0.01, value=2.0, step=0.1, format="%.2f",
            help="예: 단위 PCB당 기준 결함수 2개"
        )
        params["c_obs"] = st.sidebar.number_input(
            "관측 불량수 c", min_value=0, value=5, step=1,
            help="실제 관측된 총 불량수"
        )
        params["n_units"] = st.sidebar.number_input(
            "검사 단위 수 n", min_value=1, value=1, step=1,
            help="검사한 PCB(또는 단위) 수"
        )

    elif test_type == "levene":
        params["center"] = st.sidebar.radio(
            "Levene 중심 기준",
            ["median", "mean"],
            index=0,
            help="median: Brown-Forsythe(이상치 강건) / mean: 고전적 Levene"
        )

    return {
        "test_type": test_type,
        "alpha": alpha,
        "alternative": alternative,
        "params": params
    }


# =============================================================================
# 8. 각 검정별 메인 렌더링 함수
# =============================================================================

def render_1sample_t(df, config):
    st.markdown('<div class="section-header">📌 1-표본 t-검정 설정</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        col_name = st.selectbox("분석 컬럼 선택", df.select_dtypes(include=np.number).columns.tolist())
    with col2:
        st.info(f"유의수준: α = {config['alpha']}  |  가설: {ALT_KOR[config['alternative']]}")

    data = get_numeric_series(df, col_name).values
    if not validate_sample(data, 3, col_name):
        return

    mu0 = config["params"]["mu0"]
    alpha = config["alpha"]
    alternative = config["alternative"]

    st.markdown(f"""
    **귀무가설 H₀:** μ = {mu0}  
    **대립가설 H₁:** μ {"≠" if alternative=="two-sided" else (">" if alternative=="greater" else "<")} {mu0}  
    **n = {len(data)}**  |  **x̄ = {np.mean(data):.4f}**  |  **s = {np.std(data, ddof=1):.4f}**
    """)

    result = run_one_sample_ttest(data, mu0, alpha, alternative)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("검정통계량 t", f"{result['t_stat']:.4f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("자유도 df", f"{result['df']}"),
        (f"{int((1-alpha)*100)}% 신뢰하한", f"{result['ci_lower']:.4f}"),
        (f"{int((1-alpha)*100)}% 신뢰상한", f"{result['ci_upper']:.4f}"),
    ])

    display_interpretation(result, "1sample_t")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig1 = plot_histogram_with_normal(data,
                                           f"히스토그램 - {col_name}",
                                           mu0=mu0)
        st.pyplot(fig1)
    with col_v2:
        fig2 = plot_t_rejection_region(result["t_stat"], result["df"],
                                        alpha, alternative,
                                        f"t-분포 기각역 (df={result['df']})")
        st.pyplot(fig2)


def render_2sample_t(df, config):
    st.markdown('<div class="section-header">📌 2-표본 t-검정 설정</div>',
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_a = st.selectbox("그룹 1 컬럼", num_cols, key="g1")
    with col2:
        col_b = st.selectbox("그룹 2 컬럼", num_cols, index=min(1, len(num_cols)-1), key="g2")

    data1 = get_numeric_series(df, col_a).values
    data2 = get_numeric_series(df, col_b).values

    if not validate_sample(data1, 3, "그룹 1") or not validate_sample(data2, 3, "그룹 2"):
        return

    alpha = config["alpha"]
    alternative = config["alternative"]
    equal_var = config["params"].get("equal_var", False)

    st.markdown(f"""
    **귀무가설 H₀:** μ₁ = μ₂  
    **대립가설 H₁:** μ₁ {"≠" if alternative=="two-sided" else (">" if alternative=="greater" else "<")} μ₂  
    **그룹 1:** n={len(data1)}, x̄={np.mean(data1):.4f}, s={np.std(data1,ddof=1):.4f}  |  
    **그룹 2:** n={len(data2)}, x̄={np.mean(data2):.4f}, s={np.std(data2,ddof=1):.4f}
    """)

    result = run_two_sample_ttest(data1, data2, alpha, alternative, equal_var)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("검정통계량 t", f"{result['t_stat']:.4f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("자유도 df", f"{result['df']:.2f}"),
        ("평균 차이 Δ", f"{result['diff']:.4f}"),
        (f"{int((1-alpha)*100)}% CI (Δ)", f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"),
    ])

    display_interpretation(result, "2sample_t")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig1 = plot_boxplot_comparison(data1, data2, col_a, col_b,
                                        f"그룹 비교 ({col_a} vs {col_b})")
        st.pyplot(fig1)
    with col_v2:
        fig2 = plot_t_rejection_region(result["t_stat"], result["df"],
                                        alpha, alternative)
        st.pyplot(fig2)


def render_paired_t(df, config):
    st.markdown('<div class="section-header">📌 대응표본 t-검정 설정</div>',
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        col_before = st.selectbox("개선 전 컬럼 (Before)", num_cols, key="pb")
    with col2:
        col_after = st.selectbox("개선 후 컬럼 (After)", num_cols,
                                  index=min(1, len(num_cols)-1), key="pa")

    data_b = df[col_before].dropna().values
    data_a = df[col_after].dropna().values
    n_common = min(len(data_b), len(data_a))

    if n_common < 3:
        st.error("유효 데이터 쌍이 3개 미만입니다.")
        return

    data_b = data_b[:n_common]
    data_a = data_a[:n_common]

    alpha = config["alpha"]
    alternative = config["alternative"]

    result = run_paired_ttest(data_b, data_a, alpha, alternative)
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown(f"""
    **귀무가설 H₀:** μD = 0 (개선 전후 차이 없음)  
    **대립가설 H₁:** μD {"≠" if alternative=="two-sided" else (">" if alternative=="greater" else "<")} 0  
    **n = {result['n']}** 쌍  |  **D̄ = {result['mean_d']:.4f}**  |  **sD = {result['std_d']:.4f}**
    """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("검정통계량 t", f"{result['t_stat']:.4f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("자유도 df", f"{result['df']}"),
        ("평균 차이 D̄", f"{result['mean_d']:.4f}"),
        (f"{int((1-alpha)*100)}% CI", f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"),
    ])

    display_interpretation(result, "paired_t")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig1 = plot_boxplot_comparison(data_b, data_a, col_before, col_after,
                                        "개선 전·후 비교")
        st.pyplot(fig1)
    with col_v2:
        fig2 = plot_histogram_with_normal(result["diff_data"],
                                           "차이(D = Before - After) 분포",
                                           mu0=0)
        st.pyplot(fig2)


def render_proportion(config):
    st.markdown('<div class="section-header">📌 불량률 검정 설정</div>',
                unsafe_allow_html=True)

    alpha = config["alpha"]
    alternative = config["alternative"]
    p0 = config["params"]["p0"]
    x = int(config["params"]["x_input"])
    n = int(config["params"]["n_input"])

    if x > n:
        st.error("불량 개수(x)가 전체 샘플 수(n)보다 클 수 없습니다.")
        return
    if n < 5:
        st.warning("n이 너무 작습니다. 정규근사의 신뢰도가 낮을 수 있습니다.")

    p_hat = x / n
    st.markdown(f"""
    **귀무가설 H₀:** p = {p0:.4%}  
    **대립가설 H₁:** p {"≠" if alternative=="two-sided" else (">" if alternative=="greater" else "<")} {p0:.4%}  
    **관측값:** x={x}, n={n}, p̂={p_hat:.4%}  |  **기대 정규근사 조건:** np₀={n*p0:.1f}, n(1-p₀)={n*(1-p0):.1f}
    """)

    # 정규근사 경고
    if n * p0 < 5 or n * (1 - p0) < 5:
        st.markdown('<div class="warning-box">⚠️ np₀ 또는 n(1-p₀) < 5: 정규근사 조건 불만족. '
                    '정확 이항 검정(scipy.stats.binom_test) 사용을 권장합니다.</div>',
                    unsafe_allow_html=True)

    result = run_proportion_test(x, n, p0, alpha, alternative)
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("검정통계량 z", f"{result['z_stat']:.4f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("관측 불량률 p̂", f"{result['p_hat']:.4%}"),
        ("기준 불량률 p₀", f"{result['p0']:.4%}"),
        (f"{int((1-alpha)*100)}% CI (p)", f"[{result['ci_lower']:.4%}, {result['ci_upper']:.4%}]"),
    ])

    display_interpretation(result, "proportion")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    fig = plot_z_rejection_region(result["z_stat"], alpha, alternative,
                                   f"Z-분포 기각역 (불량률 검정)")
    st.pyplot(fig)


def render_poisson(config):
    st.markdown('<div class="section-header">📌 불량수 검정 (Poisson) 설정</div>',
                unsafe_allow_html=True)

    alpha = config["alpha"]
    alternative = config["alternative"]
    c0 = config["params"]["c0"]
    c_obs = int(config["params"]["c_obs"])
    n_units = int(config["params"]["n_units"])

    lambda_hat = c_obs / n_units
    st.markdown(f"""
    **귀무가설 H₀:** λ = {c0} (단위당 기준 불량수)  
    **대립가설 H₁:** λ {"≠" if alternative=="two-sided" else (">" if alternative=="greater" else "<")} {c0}  
    **관측:** c={c_obs}개, n={n_units}단위, λ̂={lambda_hat:.4f}  |  **기대 불량수:** {c0*n_units:.1f}
    """)

    result = run_poisson_test(c_obs, c0, n_units, alpha, alternative)
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("p-value", f"{result['p_value']:.6f}"),
        ("관측 λ̂", f"{result['lambda_hat']:.4f}"),
        ("기준 λ₀", f"{result['c0']}"),
        (f"{int((1-alpha)*100)}% CI (count)", f"[{result['ci_lower_count']:.2f}, {result['ci_upper_count']:.2f}]"),
        (f"{int((1-alpha)*100)}% CI (λ)", f"[{result['ci_lower_lambda']:.4f}, {result['ci_upper_lambda']:.4f}]"),
    ])

    display_interpretation(result, "poisson")

    # Poisson 분포 시각화
    st.markdown('<div class="section-header">📊 Poisson 분포 시각화</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    expected = c0 * n_units
    max_k = max(c_obs + 5, int(expected * 2) + 5)
    k_vals = np.arange(0, max_k + 1)
    pmf_vals = stats.poisson.pmf(k_vals, expected)

    bars = ax.bar(k_vals, pmf_vals, color="#AED6F1", edgecolor="white", label=f"Poisson(λ={expected:.2f})")

    # 기각역 색상
    if alternative == "two-sided":
        p_low = stats.poisson.cdf(c_obs, expected)
        p_high = 1 - stats.poisson.cdf(c_obs - 1, expected)
        crit_threshold = min(p_low, p_high)
        for bar, k, pmf in zip(bars, k_vals, pmf_vals):
            tail_p = min(stats.poisson.cdf(k, expected),
                         1 - stats.poisson.cdf(k - 1, expected))
            if tail_p <= alpha / 2:
                bar.set_color("#E74C3C")
    elif alternative == "greater":
        for bar, k in zip(bars, k_vals):
            if 1 - stats.poisson.cdf(k - 1, expected) <= alpha:
                bar.set_color("#E74C3C")
    else:
        for bar, k in zip(bars, k_vals):
            if stats.poisson.cdf(k, expected) <= alpha:
                bar.set_color("#E74C3C")

    ax.axvline(c_obs, color="#27AE60", lw=2.5, linestyle="--",
               label=f"관측값 c = {c_obs}")
    ax.axvline(expected, color="#F39C12", lw=2, linestyle=":",
               label=f"기대값 = {expected:.2f}")

    reject_patch = mpatches.Patch(color="#E74C3C", alpha=0.8, label="기각역")
    ax.legend(handles=list(ax.get_legend_handles_labels()[0]) + [reject_patch], fontsize=9)
    ax.set_title(f"Poisson 분포 기각역 (λ={expected:.2f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("불량수 k", fontsize=10)
    ax.set_ylabel("확률 P(X=k)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)


def render_shapiro(df, config):
    st.markdown('<div class="section-header">📌 정규성 검정 설정</div>',
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    col_name = st.selectbox("분석 컬럼 선택", num_cols)

    data = get_numeric_series(df, col_name).values
    if not validate_sample(data, 3, col_name):
        return

    alpha = config["alpha"]

    st.markdown(f"""
    **귀무가설 H₀:** 데이터가 정규분포를 따름  
    **대립가설 H₁:** 데이터가 정규분포를 따르지 않음  
    **n = {len(data)}**  |  **x̄ = {np.mean(data):.4f}**  |  **s = {np.std(data, ddof=1):.4f}**
    """)

    result = run_shapiro_wilk(data, alpha)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("W 통계량", f"{result['w_stat']:.6f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("n", f"{result['n']}"),
        ("왜도 (Skewness)", f"{result['skew']:.4f}"),
        ("첨도 (Kurtosis)", f"{result['kurtosis']:.4f}"),
    ])

    display_interpretation(result, "shapiro")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig1 = plot_histogram_with_normal(data, f"히스토그램 - {col_name}")
        st.pyplot(fig1)
    with col_v2:
        fig2 = plot_qq_plot(data, f"Q-Q Plot - {col_name}")
        st.pyplot(fig2)


def render_levene(df, config):
    st.markdown('<div class="section-header">📌 등분산 검정 설정</div>',
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.info("비교할 컬럼을 2개 이상 선택하세요 (Ctrl/Cmd 클릭으로 다중 선택)")
    selected_cols = st.multiselect(
        "분석 컬럼 선택 (2개 이상)",
        num_cols,
        default=num_cols[:min(2, len(num_cols))]
    )

    if len(selected_cols) < 2:
        st.warning("2개 이상의 컬럼을 선택해야 합니다.")
        return

    groups = [get_numeric_series(df, c).values for c in selected_cols]
    valid = all(validate_sample(g, 3, c) for g, c in zip(groups, selected_cols))
    if not valid:
        return

    alpha = config["alpha"]
    center = config["params"].get("center", "median")

    st.markdown(f"""
    **귀무가설 H₀:** σ₁² = σ₂² = ... (모든 그룹 분산 동일)  
    **대립가설 H₁:** 적어도 하나의 그룹 분산이 다름  
    **그룹 수:** {len(groups)}  |  **center = {center}**
    """)

    result = run_levene_test(*groups, alpha=alpha, center=center)
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    display_result_header(result)

    display_metric_row([
        ("F 통계량", f"{result['f_stat']:.4f}"),
        ("p-value", f"{result['p_value']:.6f}"),
        ("df1 (그룹간)", f"{result['df1']}"),
        ("df2 (그룹내)", f"{result['df2']}"),
        ("그룹 수", f"{result['n_groups']}"),
    ])

    # 그룹별 기술통계 테이블
    st.markdown('<div class="section-header">📋 그룹별 기술통계</div>',
                unsafe_allow_html=True)
    stat_df = pd.DataFrame([
        {"그룹": col,
         "n": gs["n"],
         "평균": f"{gs['mean']:.4f}",
         "표준편차": f"{gs['std']:.4f}",
         "분산": f"{gs['var']:.6f}"}
        for col, gs in zip(selected_cols, result["group_stats"])
    ])
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    display_interpretation(result, "levene")

    st.markdown('<div class="section-header">📊 시각화</div>', unsafe_allow_html=True)
    fig = plot_variance_comparison(groups, selected_cols, "그룹별 분산 비교")
    st.pyplot(fig)


# =============================================================================
# 9. 메인 앱 엔트리포인트
# =============================================================================

def main():
    # 헤더
    st.markdown('<div class="main-title">📊 가설검정 전문 분석 시스템</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**제조·품질 데이터 기반 통계적 가설검정 전문 도구** — "
        "AIAG Core Tools / IATF 16949 / 품질관리기술사 수준 분석 지원",
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # 사이드바 설정 로드
    config = render_sidebar()
    test_type = config["test_type"]

    # 검정 유형 결정: 데이터 불필요 검정
    needs_data = test_type not in ["proportion", "poisson"]

    # 데이터 업로드 섹션
    df = None
    if needs_data:
        st.markdown('<div class="section-header">📂 데이터 업로드</div>',
                    unsafe_allow_html=True)

        col_upload, col_info = st.columns([3, 2])
        with col_upload:
            uploaded = st.file_uploader(
                "CSV 또는 Excel 파일을 업로드하세요",
                type=["csv", "xlsx", "xls"],
                help="첫 번째 행은 헤더로 인식됩니다."
            )
        with col_info:
            st.markdown("""
            **지원 형식:**
            - CSV (UTF-8, EUC-KR 자동 감지)
            - Excel (.xlsx, .xls)
            
            **권장 형식:** 숫자 데이터, 열(Column) = 변수
            """)

        if uploaded:
            df = load_data(uploaded)
            if df is not None:
                with st.expander("📋 업로드된 데이터 미리보기", expanded=False):
                    st.dataframe(df.head(20), use_container_width=True)
                    st.caption(
                        f"총 {len(df)}행 × {len(df.columns)}열  |  "
                        f"수치형 컬럼: {list(df.select_dtypes(include=np.number).columns)}"
                    )
        else:
            # 샘플 데이터 제공
            st.info("📌 데이터를 업로드하거나, 아래 **샘플 데이터로 체험** 버튼을 클릭하세요.")
            if st.button("🔄 샘플 데이터 생성 (PCB 도금 두께 데이터)"):
                np.random.seed(42)
                sample_df = pd.DataFrame({
                    "두께_A공정_μm": np.random.normal(18.2, 0.8, 50),
                    "두께_B공정_μm": np.random.normal(17.8, 1.1, 50),
                    "두께_전처리_μm": np.random.normal(15.0, 0.5, 30),
                    "두께_후처리_μm": np.random.normal(18.5, 0.6, 30),
                    "저항값_Ω": np.abs(np.random.normal(100, 5, 50)),
                })
                st.session_state["sample_df"] = sample_df
                st.rerun()

            if "sample_df" in st.session_state:
                df = st.session_state["sample_df"]
                st.success("✅ 샘플 데이터 로드 완료 (PCB 도금 두께 시뮬레이션)")
                with st.expander("샘플 데이터 미리보기"):
                    st.dataframe(df.head(), use_container_width=True)

    # 분석 실행
    if needs_data and df is None:
        st.markdown('<div class="result-box">⬅️ 좌측 사이드바에서 검정 방법을 선택하고 데이터를 업로드하세요.</div>',
                    unsafe_allow_html=True)
        _render_guide()
        return

    st.markdown('<div class="section-header">🔬 분석 결과</div>', unsafe_allow_html=True)

    try:
        if test_type == "1sample_t":
            render_1sample_t(df, config)
        elif test_type == "2sample_t":
            render_2sample_t(df, config)
        elif test_type == "paired_t":
            render_paired_t(df, config)
        elif test_type == "proportion":
            render_proportion(config)
        elif test_type == "poisson":
            render_poisson(config)
        elif test_type == "shapiro":
            render_shapiro(df, config)
        elif test_type == "levene":
            render_levene(df, config)

    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
        st.exception(e)

    # 하단 정보
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption(
        "⚙️ 가설검정 전문 분석 시스템 v1.0  |  "
        "scipy.stats 기반  |  AIAG MSA 4th Edition 호환  |  "
        "향후 SPC / Cpk 분석 연계 예정"
    )


def _render_guide():
    """사용 가이드 (데이터 미업로드 시 표시)"""
    with st.expander("📖 검정 방법 가이드", expanded=True):
        st.markdown("""
        | 검정 방법 | 목적 | 사용 조건 |
        |-----------|------|-----------|
        | **1-표본 t-검정** | 공정 평균 vs 목표값 비교 | 연속형 데이터, 정규성 권장 |
        | **2-표본 t-검정** | 두 공정/설비 평균 비교 | 독립 표본, 연속형 |
        | **대응표본 t-검정** | 개선 전·후 효과 검정 | 동일 대상 반복 측정 |
        | **불량률 검정** | 관측 불량률 vs 기준 비교 | 이항 데이터, np≥5 권장 |
        | **불량수 검정** | 결함수 vs 기준 비교 | Poisson 분포 가정 |
        | **정규성 검정** | 데이터 정규분포 여부 | n ≤ 5,000 권장 |
        | **등분산 검정** | 그룹 간 분산 동질성 | 2개 이상 그룹 |
        """)


# =============================================================================
# 10. 앱 실행
# =============================================================================

if __name__ == "__main__":
    main()
