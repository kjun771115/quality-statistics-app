"""
============================================================
  공정능력 분석 전문가 도구 (Process Capability Analyzer)
  버전: 1.0.0
  개발 목적: 제조 공정 데이터 기반 전문가용 공정능력 분석
  적용 표준: AIAG Core Tools (SPC), ISO/TS 16949 기반
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import shapiro, norm
import io
import base64
import warnings
import os
import platform

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. 페이지 설정 및 스타일
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="공정능력 분석 도구",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a3a5c;
        border-bottom: 3px solid #2196F3; padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 700; color: #1565C0;
        background: #E3F2FD; padding: 0.4rem 0.8rem;
        border-left: 4px solid #1565C0; border-radius: 2px;
        margin: 1.2rem 0 0.8rem 0;
    }
    .metric-card {
        background: #F8F9FA; border: 1px solid #DEE2E6;
        border-radius: 8px; padding: 0.8rem 1rem;
        text-align: center;
    }
    .verdict-excellent { background:#E8F5E9; border-left:5px solid #2E7D32; padding:1rem; border-radius:4px; }
    .verdict-good      { background:#E3F2FD; border-left:5px solid #1565C0; padding:1rem; border-radius:4px; }
    .verdict-warning   { background:#FFF8E1; border-left:5px solid #F9A825; padding:1rem; border-radius:4px; }
    .verdict-danger    { background:#FFEBEE; border-left:5px solid #C62828; padding:1rem; border-radius:4px; }
    .stDownloadButton>button { width:100%; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 2. 한글 폰트 설정 (플랫폼별 자동 탐지)
# ──────────────────────────────────────────────
def setup_korean_font():
    """운영 체제별 한글 폰트를 자동 탐지하여 matplotlib에 적용"""
    candidates = []
    system = platform.system()

    if system == "Windows":
        candidates = ["Malgun Gothic", "NanumGothic", "Gulim", "Dotum"]
    elif system == "Darwin":  # macOS
        candidates = ["AppleGothic", "NanumGothic", "Arial Unicode MS"]
    else:  # Linux / Streamlit Cloud
        linux_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/nanum/NanumGothic.ttf",
            "/usr/local/share/fonts/NanumGothic.ttf",
        ]
        for p in linux_paths:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
        candidates = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Noto Sans CJK KR",
                       "DejaVu Sans"]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name

    # 폴백: 기본 폰트 사용 (영문)
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

FONT_NAME = setup_korean_font()


# ──────────────────────────────────────────────
# 3. 데이터 처리 함수
# ──────────────────────────────────────────────
def load_data(uploaded_file) -> pd.DataFrame:
    """CSV 또는 Excel 파일을 DataFrame으로 로드"""
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            # 인코딩 자동 탐지 (UTF-8, CP949 순서로 시도)
            for enc in ["utf-8", "cp949", "euc-kr", "utf-8-sig"]:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc)
                except (UnicodeDecodeError, Exception):
                    continue
            raise ValueError("CSV 파일 인코딩을 인식할 수 없습니다.")
        elif fname.endswith((".xls", ".xlsx", ".xlsm")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. (CSV, Excel만 가능)")
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return pd.DataFrame()


def remove_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """IQR 기반 이상치 제거"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series[(series >= q1 - factor * iqr) & (series <= q3 + factor * iqr)]


def remove_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score 기반 이상치 제거"""
    z = np.abs(stats.zscore(series.dropna()))
    return series.dropna()[z < threshold]


# ──────────────────────────────────────────────
# 4. 통계 분석 함수
# ──────────────────────────────────────────────
def calc_basic_stats(data: pd.Series) -> dict:
    """기초 통계량 계산"""
    return {
        "N": len(data),
        "Mean (μ)": data.mean(),
        "Std Dev (σ)": data.std(ddof=1),        # 표본 표준편차
        "Std Dev (σ_p)": data.std(ddof=0),      # 모 표준편차 (Pp/Ppk용)
        "Min": data.min(),
        "Max": data.max(),
        "Median": data.median(),
        "Skewness": data.skew(),
        "Kurtosis": data.kurt(),
        "Range": data.max() - data.min(),
    }


def calc_capability(data: pd.Series, lsl: float, usl: float) -> dict:
    """
    공정능력지수 계산
    - Cp/Cpk: 표본 표준편차 기반 (단기 공정능력, 관리 상태 가정)
    - Pp/Ppk: 전체(모) 표준편차 기반 (장기 공정능력, 실제 성능)
    """
    mu = data.mean()
    sigma_s = data.std(ddof=1)   # 표본 표준편차
    sigma_p = data.std(ddof=0)   # 전체 표준편차

    def safe_div(a, b):
        return a / b if b != 0 else float("nan")

    spread = usl - lsl

    # Cp, Cpk (단기 능력)
    cp  = safe_div(spread, 6 * sigma_s)
    cpu = safe_div(usl - mu, 3 * sigma_s)
    cpl = safe_div(mu - lsl, 3 * sigma_s)
    cpk = min(cpu, cpl)

    # Pp, Ppk (장기 성능)
    pp  = safe_div(spread, 6 * sigma_p)
    ppu = safe_div(usl - mu, 3 * sigma_p)
    ppl = safe_div(mu - lsl, 3 * sigma_p)
    ppk = min(ppu, ppl)

    # 규격 이탈 확률 (정규 분포 가정)
    z_usl = safe_div(usl - mu, sigma_s)
    z_lsl = safe_div(mu - lsl, sigma_s)
    ppm_out = (1 - (norm.cdf(z_usl) - norm.cdf(-z_lsl))) * 1_000_000

    return {
        "Cp": cp, "CPU": cpu, "CPL": cpl, "Cpk": cpk,
        "Pp": pp, "PPU": ppu, "PPL": ppl, "Ppk": ppk,
        "PPM (예상)": ppm_out,
    }


def run_normality_test(data: pd.Series) -> dict:
    """
    정규성 검정 (Shapiro-Wilk)
    - N ≤ 5000: Shapiro-Wilk (높은 검정력)
    - N > 5000: Kolmogorov-Smirnov (대표본 대응)
    """
    n = len(data)
    if n < 3:
        return {"test": "N/A", "statistic": None, "p_value": None, "is_normal": None}
    if n <= 5000:
        stat, p = shapiro(data)
        test_name = "Shapiro-Wilk"
    else:
        stat, p = stats.kstest(data, "norm",
                               args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p,
        "is_normal": p > 0.05,
    }


# ──────────────────────────────────────────────
# 5. 시각화 함수
# ──────────────────────────────────────────────
def fig_to_bytes(fig: plt.Figure) -> bytes:
    """matplotlib Figure를 PNG 바이트로 변환"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def plot_histogram(data: pd.Series, lsl: float, usl: float,
                   target: float = None, col_name: str = "") -> plt.Figure:
    """히스토그램 + 정규분포 곡선 + 규격 한계선"""
    fig, ax = plt.subplots(figsize=(9, 5))

    mu, sigma = data.mean(), data.std(ddof=1)
    n_bins = max(10, min(50, int(np.sqrt(len(data)))))

    # 히스토그램
    counts, bin_edges, patches = ax.hist(
        data, bins=n_bins, density=True,
        color="#90CAF9", edgecolor="#1565C0", linewidth=0.6, alpha=0.85,
        label="데이터 분포"
    )

    # 정규분포 곡선 overlay
    x_range = np.linspace(
        min(data.min(), lsl - 2 * sigma),
        max(data.max(), usl + 2 * sigma), 300
    )
    ax.plot(x_range, norm.pdf(x_range, mu, sigma),
            "r-", linewidth=2.2, label=f"정규분포 N({mu:.3f}, {sigma:.3f}²)")

    # 규격 한계선
    ax.axvline(lsl, color="#C62828", linestyle="--", linewidth=2,
               label=f"LSL = {lsl:.4g}")
    ax.axvline(usl, color="#C62828", linestyle="--", linewidth=2,
               label=f"USL = {usl:.4g}")
    if target is not None:
        ax.axvline(target, color="#2E7D32", linestyle="-.", linewidth=1.8,
                   label=f"Target = {target:.4g}")

    # 평균선
    ax.axvline(mu, color="#F57F17", linestyle=":", linewidth=2,
               label=f"μ = {mu:.4g}")

    # 규격 밖 영역 음영
    y_max = ax.get_ylim()[1]
    x_fill = np.linspace(x_range[0], x_range[-1], 500)
    y_fill = norm.pdf(x_fill, mu, sigma)
    ax.fill_between(x_fill, y_fill, where=(x_fill < lsl),
                    color="#FFCDD2", alpha=0.5)
    ax.fill_between(x_fill, y_fill, where=(x_fill > usl),
                    color="#FFCDD2", alpha=0.5)

    ax.set_title(f"히스토그램 + 정규분포 곡선: {col_name}", fontsize=14, fontweight="bold",
                 pad=12)
    ax.set_xlabel("측정값", fontsize=11)
    ax.set_ylabel("밀도", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_boxplot(data: pd.Series, lsl: float, usl: float,
                 target: float = None, col_name: str = "") -> plt.Figure:
    """Box Plot + 규격 한계선"""
    fig, ax = plt.subplots(figsize=(6, 5))

    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#C62828", linewidth=2.5),
                    flierprops=dict(marker="o", color="#EF9A9A",
                                   markerfacecolor="#EF9A9A", markersize=5),
                    boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
                    whiskerprops=dict(color="#1565C0", linewidth=1.5),
                    capprops=dict(color="#1565C0", linewidth=2))

    ax.axhline(lsl, color="#C62828", linestyle="--", linewidth=2,
               label=f"LSL = {lsl:.4g}")
    ax.axhline(usl, color="#C62828", linestyle="--", linewidth=2,
               label=f"USL = {usl:.4g}")
    if target is not None:
        ax.axhline(target, color="#2E7D32", linestyle="-.", linewidth=1.8,
                   label=f"Target = {target:.4g}")

    ax.set_title(f"Box Plot: {col_name}", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("측정값", fontsize=11)
    ax.set_xticks([])
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 통계 주석
    stats_text = (
        f"μ = {data.mean():.4g}\n"
        f"σ = {data.std(ddof=1):.4g}\n"
        f"N = {len(data)}"
    )
    ax.text(1.38, data.mean(), stats_text,
            fontsize=9, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    return fig


def plot_qqplot(data: pd.Series, col_name: str = "") -> plt.Figure:
    """Q-Q Plot (정규성 시각적 확인)"""
    fig, ax = plt.subplots(figsize=(6, 5))

    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax.scatter(osm, osr, color="#42A5F5", edgecolors="#1565C0",
               s=25, alpha=0.75, label="데이터 포인트")
    ref_line = np.linspace(min(osm), max(osm), 100)
    ax.plot(ref_line, slope * ref_line + intercept,
            "r-", linewidth=2, label=f"기준선 (R²={r**2:.4f})")

    ax.set_title(f"Q-Q Plot (정규성 검토): {col_name}", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlabel("이론 분위수 (Theoretical Quantiles)", fontsize=10)
    ax.set_ylabel("표본 분위수 (Sample Quantiles)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_capability_dashboard(data: pd.Series, lsl: float, usl: float,
                               cap: dict, col_name: str = "") -> plt.Figure:
    """공정능력 게이지 대시보드 (Cp/Cpk/Pp/Ppk 시각화)"""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    indicators = [
        ("Cp",  cap["Cp"],  "#42A5F5"),
        ("Cpk", cap["Cpk"], "#1565C0"),
        ("Pp",  cap["Pp"],  "#66BB6A"),
        ("Ppk", cap["Ppk"], "#2E7D32"),
    ]

    for ax, (name, val, color) in zip(axes, indicators):
        if np.isnan(val):
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
            ax.set_title(name, fontsize=14, fontweight="bold")
            ax.axis("off")
            continue

        # 게이지 막대
        max_val = max(2.0, val * 1.2)
        bars = [1.0, 1.33, 1.67, max_val]
        bar_colors = ["#FFCDD2", "#FFF9C4", "#C8E6C9", "#E8F5E9"]

        left = 0
        for bv, bc in zip(bars, bar_colors):
            ax.barh(0, bv - left, left=left, height=0.5,
                    color=bc, edgecolor="grey", linewidth=0.4)
            left = bv

        # 현재값 마커
        ax.axvline(val, color=color, linewidth=3.5)
        ax.text(val, 0.42, f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

        # 기준선
        for xv, lbl in [(1.0, "1.00"), (1.33, "1.33"), (1.67, "1.67")]:
            ax.axvline(xv, color="grey", linestyle=":", linewidth=1)
            ax.text(xv, -0.38, lbl, ha="center", va="top", fontsize=7.5,
                    color="grey")

        ax.set_xlim(0, max_val)
        ax.set_ylim(-0.5, 0.7)
        ax.set_yticks([])
        ax.set_title(name, fontsize=13, fontweight="bold", pad=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"공정능력 지수 대시보드: {col_name}",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 6. 판정 / 해석 함수
# ──────────────────────────────────────────────
def get_cpk_verdict(cpk: float) -> tuple[str, str, str]:
    """
    Cpk 값에 따른 판정 등급, 설명, CSS 클래스 반환
    기준: AIAG SPC 매뉴얼 (2nd Edition) 권고 기준
    """
    if np.isnan(cpk):
        return "판정 불가", "표준편차가 0이거나 데이터 부족으로 계산 불가.", "verdict-warning"
    if cpk >= 1.67:
        return "매우 우수 (Excellent)", "공정은 규격을 충분히 만족합니다. 현 수준 유지 권고.", "verdict-excellent"
    elif cpk >= 1.33:
        return "양호 (Good)", "공정능력이 적정 수준입니다. 지속적 모니터링을 권고합니다.", "verdict-good"
    elif cpk >= 1.00:
        return "개선 필요 (Marginal)", "공정능력이 경계 수준입니다. 원인 분석 및 개선 조치가 필요합니다.", "verdict-warning"
    else:
        return "불량 위험 (Poor)", "공정능력 불충분. 즉각적인 원인 분석 및 시정조치가 필요합니다.", "verdict-danger"


def generate_report(stats_d: dict, cap: dict, norm_d: dict,
                    lsl: float, usl: float, col_name: str) -> str:
    """
    자동 해석 리포트 텍스트 생성
    품질 컨설턴트 수준의 서술 형태
    """
    cpk = cap["Cpk"]
    ppk = cap["Ppk"]
    mu  = stats_d["Mean (μ)"]
    sig = stats_d["Std Dev (σ)"]
    n   = stats_d["N"]
    is_normal = norm_d.get("is_normal")
    p_val = norm_d.get("p_value")
    test_name = norm_d.get("test", "N/A")

    verdict, _, _ = get_cpk_verdict(cpk)

    lines = [
        f"■ 분석 대상: {col_name}  |  샘플 수: {n}개  |  LSL: {lsl:.4g}, USL: {usl:.4g}",
        "",
        "▶ [정규성 검정]",
    ]

    if is_normal is None:
        lines.append("  · 데이터 수 부족으로 정규성 검정을 수행할 수 없습니다.")
    else:
        norm_result = "만족" if is_normal else "불만족"
        norm_advice = (
            "정규분포 가정 하에 공정능력 지수의 신뢰성이 확보됩니다."
            if is_normal else
            "비정규 분포로 판단됩니다. 변환(Box-Cox, Johnson 등) 적용 또는 비모수 공정능력 지수(Cnpk) 활용을 검토하십시오."
        )
        lines += [
            f"  · 검정 방법: {test_name}",
            f"  · 검정 통계량: {norm_d['statistic']:.5f},  p-value: {p_val:.5f}",
            f"  · 결론: 공정은 정규성을 {norm_result}합니다. (유의수준 α=0.05)",
            f"  · 해석: {norm_advice}",
        ]

    lines += [
        "",
        "▶ [공정능력 분석]",
        f"  · 공정 평균(μ) = {mu:.4g},  표준편차(σ) = {sig:.4g}",
        f"  · Cp = {cap['Cp']:.3f}  (공정 산포 대비 규격 폭 비율)",
        f"  · Cpk = {cpk:.3f}  (공정 중심 이탈 고려 단기 능력)",
        f"  · Pp = {cap['Pp']:.3f}  (장기 공정 산포 기반 규격 여유)",
        f"  · Ppk = {ppk:.3f}  (장기 실제 성능 지수)",
        f"  · 판정: {verdict}",
        "",
        "▶ [종합 해석]",
    ]

    # Cp vs Cpk 비교 해석
    if not np.isnan(cap["Cp"]) and not np.isnan(cpk):
        diff = cap["Cp"] - cpk
        if diff > 0.2:
            lines.append(f"  · Cp({cap['Cp']:.3f}) > Cpk({cpk:.3f}) 격차가 크므로 공정 중심이 규격 중앙에서 편향되어 있습니다.")
            lines.append(f"    → 공정 중심(평균)을 규격 중앙값({(lsl+usl)/2:.4g})에 맞추는 조치를 권고합니다.")
        else:
            lines.append(f"  · Cp와 Cpk 값이 유사합니다. 공정 중심은 규격 중앙에 비교적 정렬되어 있습니다.")

    # PPM 해석
    ppm = cap.get("PPM (예상)", float("nan"))
    if not np.isnan(ppm):
        lines.append(f"  · 추정 불량률: {ppm:,.1f} PPM ({ppm/10000:.4f}%)")
        if ppm < 3.4:
            lines.append("    → Six Sigma 수준의 불량률입니다.")
        elif ppm < 6210:
            lines.append("    → 4σ 수준 이상으로 관리 가능한 범위입니다.")
        else:
            lines.append("    → 불량률 감소를 위한 공정 개선이 시급합니다.")

    # 최종 권고
    lines += ["", "▶ [조치 권고]"]
    if cpk >= 1.67:
        lines.append("  · 현재 공정 수준을 유지하며 정기적인 모니터링(SPC 관리도)을 지속하십시오.")
    elif cpk >= 1.33:
        lines.append("  · 공정 변동 요인을 지속 관리하고, 산포 감소 활동을 추진하십시오.")
    elif cpk >= 1.00:
        lines.append("  · 잠재적 불량 위험이 있습니다. 4M 분석(인/기계/재료/방법)을 실시하고 공정 개선을 추진하십시오.")
    else:
        lines.append("  · 즉각적인 원인 분석(특성 요인도, FMEA 재검토)과 시정 및 예방 조치(CAPA)를 수립하십시오.")
        lines.append("  · 필요시 공정 중단 및 전수 검사를 검토하십시오.")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 7. 결과 CSV 생성
# ──────────────────────────────────────────────
def build_result_csv(results: list[dict]) -> bytes:
    """분석 결과를 CSV 바이트로 변환 (다중 컬럼 지원)"""
    rows = []
    for r in results:
        row = {"컬럼명": r["col"]}
        row.update(r["stats"])
        row.update(r["cap"])
        row["정규성_검정"] = r["norm"]["test"]
        row["p-value"] = r["norm"].get("p_value", "N/A")
        row["정규성_여부"] = "만족" if r["norm"].get("is_normal") else "불만족"
        row["판정"] = get_cpk_verdict(r["cap"]["Cpk"])[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ──────────────────────────────────────────────
# 8. 단일 컬럼 분석 렌더링
# ──────────────────────────────────────────────
def render_analysis(col_name: str, raw_data: pd.Series,
                    lsl: float, usl: float, target: float | None,
                    remove_missing: bool, outlier_method: str,
                    log_scale: bool) -> dict:
    """
    하나의 컬럼에 대한 전체 분석 수행 및 Streamlit 렌더링
    Returns: result dict (CSV 저장용)
    """
    st.markdown(f'<div class="section-header">📌 분석 컬럼: {col_name}</div>',
                unsafe_allow_html=True)

    # ── 데이터 전처리 ──
    data = raw_data.copy()

    if remove_missing:
        before = len(data)
        data = data.dropna()
        st.caption(f"결측치 제거: {before - len(data)}개 제거 → {len(data)}개 잔여")

    # 숫자 변환
    data = pd.to_numeric(data, errors="coerce").dropna()

    if len(data) < 5:
        st.error(f"'{col_name}': 분석에 필요한 데이터가 부족합니다 (최소 5개 이상 필요, 현재 {len(data)}개).")
        return {}

    # 이상치 제거
    if outlier_method == "IQR":
        data_clean = remove_outliers_iqr(data)
        removed = len(data) - len(data_clean)
        if removed > 0:
            st.caption(f"IQR 이상치 제거: {removed}개 제거 → {len(data_clean)}개 잔여")
        data = data_clean
    elif outlier_method == "Z-score":
        data_clean = remove_outliers_zscore(data)
        removed = len(data) - len(data_clean)
        if removed > 0:
            st.caption(f"Z-score 이상치 제거: {removed}개 제거 → {len(data_clean)}개 잔여")
        data = data_clean

    if log_scale and (data <= 0).any():
        st.warning("로그 스케일 옵션: 0 이하 값이 포함되어 있어 로그 변환을 적용할 수 없습니다.")
        log_scale = False

    if log_scale:
        data = np.log(data)
        st.caption("📐 로그(ln) 변환이 적용된 데이터로 분석합니다.")

    if len(data) < 5:
        st.error(f"전처리 후 데이터가 부족합니다 ({len(data)}개).")
        return {}

    # ── 통계 계산 ──
    stats_d = calc_basic_stats(data)
    cap     = calc_capability(data, lsl, usl)
    norm_d  = run_normality_test(data)
    verdict, verdict_desc, verdict_css = get_cpk_verdict(cap["Cpk"])

    # ─────────────────────────────────────────
    # 섹션 1: 데이터 요약
    # ─────────────────────────────────────────
    st.markdown('<div class="section-header">① 데이터 요약 통계</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("샘플 수 (N)", f"{stats_d['N']}개", None),
        ("평균 (μ)", f"{stats_d['Mean (μ)']:.4g}", None),
        ("표준편차 (σ)", f"{stats_d['Std Dev (σ)']:.4g}", None),
        ("최솟값", f"{stats_d['Min']:.4g}", None),
        ("최댓값", f"{stats_d['Max']:.4g}", None),
    ]
    for col_ui, (label, val, delta) in zip(cols, metrics):
        col_ui.metric(label, val, delta)

    extra_cols = st.columns(4)
    extra_metrics = [
        ("중앙값", f"{stats_d['Median']:.4g}"),
        ("왜도 (Skewness)", f"{stats_d['Skewness']:.4f}"),
        ("첨도 (Kurtosis)", f"{stats_d['Kurtosis']:.4f}"),
        ("범위 (Range)", f"{stats_d['Range']:.4g}"),
    ]
    for col_ui, (label, val) in zip(extra_cols, extra_metrics):
        col_ui.metric(label, val)

    # ─────────────────────────────────────────
    # 섹션 2: 공정능력 결과
    # ─────────────────────────────────────────
    st.markdown('<div class="section-header">② 공정능력 지수 (Process Capability Indices)</div>',
                unsafe_allow_html=True)

    # 판정 배너
    st.markdown(f'<div class="{verdict_css}"><b>판정: {verdict}</b> — {verdict_desc}</div>',
                unsafe_allow_html=True)
    st.write("")

    # 능력 지수 테이블
    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown("**단기 공정능력 (표본 σ 기반)**")
        df_short = pd.DataFrame({
            "지수": ["Cp", "CPU", "CPL", "Cpk"],
            "값": [f"{cap[k]:.4f}" if not np.isnan(cap[k]) else "N/A"
                   for k in ["Cp", "CPU", "CPL", "Cpk"]],
            "기준(≥1.33)": ["✅" if not np.isnan(cap[k]) and cap[k] >= 1.33 else "❌"
                            for k in ["Cp", "CPU", "CPL", "Cpk"]],
        })
        st.dataframe(df_short, hide_index=True, use_container_width=True)

    with cap_col2:
        st.markdown("**장기 공정성능 (전체 σ 기반)**")
        df_long = pd.DataFrame({
            "지수": ["Pp", "PPU", "PPL", "Ppk"],
            "값": [f"{cap[k]:.4f}" if not np.isnan(cap[k]) else "N/A"
                   for k in ["Pp", "PPU", "PPL", "Ppk"]],
            "기준(≥1.33)": ["✅" if not np.isnan(cap[k]) and cap[k] >= 1.33 else "❌"
                            for k in ["Pp", "PPU", "PPL", "Ppk"]],
        })
        st.dataframe(df_long, hide_index=True, use_container_width=True)

    ppm_val = cap.get("PPM (예상)", float("nan"))
    if not np.isnan(ppm_val):
        st.info(f"📌 추정 불량률: **{ppm_val:,.1f} PPM** ({ppm_val / 10000:.4f}%)")

    # 정규성 검정 결과
    st.markdown("**정규성 검정 결과**")
    if norm_d["p_value"] is not None:
        norm_cols = st.columns(3)
        norm_cols[0].metric("검정 방법", norm_d["test"])
        norm_cols[1].metric("검정 통계량", f"{norm_d['statistic']:.5f}")
        norm_cols[2].metric(
            "p-value",
            f"{norm_d['p_value']:.5f}",
            delta="정규성 만족" if norm_d["is_normal"] else "정규성 불만족",
            delta_color="normal" if norm_d["is_normal"] else "inverse",
        )
    else:
        st.warning("정규성 검정 결과를 산출할 수 없습니다.")

    # 게이지 대시보드
    st.markdown('<div class="section-header">③ 그래프</div>', unsafe_allow_html=True)

    fig_dashboard = plot_capability_dashboard(data, lsl, usl, cap, col_name)
    st.pyplot(fig_dashboard)
    st.download_button(
        "💾 게이지 대시보드 저장 (PNG)",
        data=fig_to_bytes(fig_dashboard),
        file_name=f"{col_name}_capability_dashboard.png",
        mime="image/png",
        key=f"dl_dash_{col_name}",
    )
    plt.close(fig_dashboard)

    # 히스토그램
    fig_hist = plot_histogram(data, lsl, usl, target, col_name)
    st.pyplot(fig_hist)
    st.download_button(
        "💾 히스토그램 저장 (PNG)",
        data=fig_to_bytes(fig_hist),
        file_name=f"{col_name}_histogram.png",
        mime="image/png",
        key=f"dl_hist_{col_name}",
    )
    plt.close(fig_hist)

    # Box Plot + Q-Q Plot (나란히)
    bp_col, qq_col = st.columns(2)
    with bp_col:
        fig_box = plot_boxplot(data, lsl, usl, target, col_name)
        st.pyplot(fig_box)
        st.download_button(
            "💾 Box Plot 저장",
            data=fig_to_bytes(fig_box),
            file_name=f"{col_name}_boxplot.png",
            mime="image/png",
            key=f"dl_box_{col_name}",
        )
        plt.close(fig_box)

    with qq_col:
        fig_qq = plot_qqplot(data, col_name)
        st.pyplot(fig_qq)
        st.download_button(
            "💾 Q-Q Plot 저장",
            data=fig_to_bytes(fig_qq),
            file_name=f"{col_name}_qqplot.png",
            mime="image/png",
            key=f"dl_qq_{col_name}",
        )
        plt.close(fig_qq)

    # ─────────────────────────────────────────
    # 섹션 4: 해석 리포트
    # ─────────────────────────────────────────
    st.markdown('<div class="section-header">④ 해석 리포트 (Interpretation Report)</div>',
                unsafe_allow_html=True)

    report_text = generate_report(stats_d, cap, norm_d, lsl, usl, col_name)
    st.text_area("", value=report_text, height=340, key=f"report_{col_name}")

    st.download_button(
        "💾 리포트 텍스트 저장 (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"{col_name}_report.txt",
        mime="text/plain",
        key=f"dl_report_{col_name}",
    )

    st.divider()

    return {"col": col_name, "stats": stats_d, "cap": cap, "norm": norm_d}


# ──────────────────────────────────────────────
# 9. 메인 앱
# ──────────────────────────────────────────────
def main():
    # 헤더
    st.markdown('<div class="main-header">📊 공정능력 분석 전문가 도구<br>'
                '<span style="font-size:1rem;color:#546E7A;">Process Capability Analyzer — AIAG SPC 기반</span>'
                '</div>', unsafe_allow_html=True)

    # ──── 사이드바 ────
    with st.sidebar:
        st.header("⚙️ 분석 설정")
        st.markdown("---")

        # 파일 업로드
        st.subheader("1️⃣ 데이터 업로드")
        uploaded = st.file_uploader(
            "CSV 또는 Excel 파일",
            type=["csv", "xlsx", "xls", "xlsm"],
            help="CSV(UTF-8/CP949) 또는 Excel 형식을 지원합니다."
        )

        st.markdown("---")
        st.subheader("2️⃣ 규격 입력")
        lsl_inp = st.number_input("LSL (하한 규격)", value=0.0,
                                   format="%.6g", key="lsl")
        usl_inp = st.number_input("USL (상한 규격)", value=1.0,
                                   format="%.6g", key="usl")
        use_target = st.checkbox("목표값(Target) 사용", value=False)
        target_inp = None
        if use_target:
            target_inp = st.number_input(
                "Target", value=(lsl_inp + usl_inp) / 2,
                format="%.6g", key="target"
            )

        st.markdown("---")
        st.subheader("3️⃣ 데이터 전처리")
        remove_missing = st.checkbox("결측치 자동 제거", value=True)
        outlier_method = st.selectbox(
            "이상치 제거 방법",
            ["없음", "IQR", "Z-score"],
            help="IQR: 사분위수 기반 / Z-score: 평균으로부터 3σ 이탈 제거"
        )
        log_scale = st.checkbox(
            "로그 스케일 변환 (ln)",
            value=False,
            help="데이터가 로그 정규 분포를 따를 경우 사용"
        )

        st.markdown("---")
        st.caption(f"폰트: {FONT_NAME}")
        st.caption("© 2025 Process Capability Analyzer v1.0")

    # ──── 메인 영역 ────
    if uploaded is None:
        # 시작 화면
        st.info("👈 왼쪽 사이드바에서 CSV 또는 Excel 파일을 업로드하고, 규격(LSL/USL)을 입력하세요.")

        with st.expander("📋 사용 방법 안내", expanded=True):
            st.markdown("""
**1단계 — 파일 업로드**
- CSV (UTF-8, CP949) 또는 Excel 파일을 업로드합니다.
- 각 컬럼이 측정값을 나타내는 정형 데이터를 준비하세요.

**2단계 — 규격 입력**
- LSL(하한 규격)과 USL(상한 규격)을 입력합니다.
- 선택적으로 목표값(Target)을 설정할 수 있습니다.

**3단계 — 분석 컬럼 선택**
- 파일 로드 후 분석할 컬럼(들)을 선택합니다.
- 다중 선택 시 각 컬럼별 반복 분석이 수행됩니다.

**4단계 — 결과 확인**
- 기초 통계, 공정능력 지수(Cp/Cpk/Pp/Ppk), 시각화, 해석 리포트가 자동 생성됩니다.
- 그래프 및 결과 CSV를 개별 다운로드할 수 있습니다.

---
**공정능력 판정 기준 (AIAG SPC 매뉴얼)**

| Cpk 범위 | 판정 |
|---|---|
| ≥ 1.67 | 매우 우수 (Excellent) |
| 1.33 ~ 1.67 | 양호 (Good) |
| 1.00 ~ 1.33 | 개선 필요 (Marginal) |
| < 1.00 | 불량 위험 (Poor) |
            """)
        return

    # ──── 데이터 로드 ────
    df = load_data(uploaded)
    if df.empty:
        st.error("데이터 로드에 실패했습니다. 파일 형식 및 인코딩을 확인하세요.")
        return

    # 숫자형 컬럼만 선택 가능하도록 필터링
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("수치형 컬럼이 없습니다. 데이터를 확인하세요.")
        return

    # 컬럼 선택
    st.markdown("### 📂 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"총 {len(df):,}행 × {len(df.columns)}열  |  수치형 컬럼: {len(numeric_cols)}개")

    selected_cols = st.multiselect(
        "🔍 분석할 컬럼 선택 (복수 선택 가능)",
        options=numeric_cols,
        default=numeric_cols[:1],
        help="수치형 컬럼만 표시됩니다."
    )

    if not selected_cols:
        st.warning("분석할 컬럼을 하나 이상 선택하세요.")
        return

    # 규격 유효성 검사
    if lsl_inp >= usl_inp:
        st.error("LSL이 USL보다 크거나 같습니다. 올바른 규격을 입력하세요.")
        return

    # 분석 실행 버튼
    run_btn = st.button("▶ 분석 실행", type="primary", use_container_width=True)

    if not run_btn:
        return

    st.markdown("---")
    st.markdown("## 🔬 분석 결과")

    all_results = []

    for col_name in selected_cols:
        try:
            result = render_analysis(
                col_name=col_name,
                raw_data=df[col_name],
                lsl=lsl_inp,
                usl=usl_inp,
                target=target_inp,
                remove_missing=remove_missing,
                outlier_method=outlier_method,
                log_scale=log_scale,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            st.error(f"'{col_name}' 분석 중 오류 발생: {e}")

    # ──── 전체 결과 CSV 다운로드 ────
    if all_results:
        st.markdown("---")
        st.markdown("### 💾 전체 분석 결과 다운로드")
        csv_bytes = build_result_csv(all_results)
        st.download_button(
            label="📥 전체 분석 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="process_capability_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("CSV는 BOM 포함 UTF-8 형식으로 저장됩니다. (Excel 한글 호환)")


# ──────────────────────────────────────────────
# 10. 엔트리포인트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()

