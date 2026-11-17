# -*- coding: utf-8 -*-
"""
Q4 케이스 유형 군집 + 개선안 효과 시뮬레이션 + A/B 테스트 설계
준비: q3_model.py 실행 후 output/case_model.pkl, log_clean.pkl 이 있어야 함
실행: python3 q4_cluster_ab_sim.py

  1. 군집(K-Means)   : 단계별 시간 패턴으로 케이스를 묶어 '어디서 시간이 새는 유형인지' 구분
  2. 효과 시뮬레이션 : 점검 대기를 줄이면 전체 소요시간이 얼마나 줄어드는지 계산 (개선안의 크기)
  3. A/B 테스트 설계 : '점검 지연 알림'을 실험한다면 몇 건, 며칠이 필요한지 검정력 시뮬레이션
"""
import os, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 250, "display.max_columns", 30)
SEED = 42
rng = np.random.default_rng(SEED)

rep = pd.read_pickle(f"{OUT_DIR}/case_model.pkl")
log = pd.read_pickle(f"{OUT_DIR}/log_clean.pkl")

# ═════════ 1. 단계별 시간 → K-Means 군집 ═════════
ev = log.dropna(subset=["timestamp"])
first = ev.groupby(["caseID", "taskID", "eventtype"])["timestamp"].min().unstack([1, 2])
last = ev.groupby(["caseID", "taskID", "eventtype"])["timestamp"].max().unstack([1, 2])
REPAIRS = ["InternRepair", "ExternRepair", "ImmediateRepair"]
rep_start = pd.concat([first[(t, "start")] for t in REPAIRS if (t, "start") in first], axis=1).min(axis=1)

h = lambda a, b: (b - a).dt.total_seconds() / 3600
st = pd.DataFrame({
    "접수_h":      h(first[("MakeTicket", "start")], first[("InformClientSurvey", "complete")]),
    "점검대기_h":   h(first[("InformClientSurvey", "complete")], first[("Survey", "start")]),
    "점검작업_h":   h(first[("Survey", "start")], first[("Survey", "complete")]),
    "수리대기_h":   h(first[("Survey", "complete")], rep_start),
    "수리작업_h":   h(rep_start, first[("RepairReady", "complete")]),
    "마감_h":      h(first[("RepairReady", "complete")], last[("TicketReady", "complete")]),
}).reindex(rep.index)
st = st[(st >= 0).all(axis=1)].dropna()                       # 시각 결측·역전 케이스 제외
print(f"[군집 대상] 단계 시간이 모두 계산되는 케이스 {len(st)} / {len(rep)}건")

X = StandardScaler().fit_transform(np.log1p(st))              # 오른쪽 꼬리가 길어 로그 변환
sil = {k: silhouette_score(X, KMeans(k, n_init=20, random_state=SEED).fit_predict(X)) for k in range(2, 8)}
print("[실루엣 점수]", {k: round(v, 3) for k, v in sil.items()})
K = max(sil, key=sil.get)
km = KMeans(K, n_init=20, random_state=SEED).fit(X)
st["cluster"] = km.labels_
st = st.join(rep[["RepairType", "duration_d", "long"]])

prof = st.groupby("cluster").agg(cases=("duration_d", "size"), **{c: (c, "median") for c in st.columns[:6]},
                                 소요일_중앙=("duration_d", "median"), 일주일초과=("long", "mean")).round(2)
prof["비중"] = (prof["cases"] / prof["cases"].sum()).round(3)
mix = pd.crosstab(st["cluster"], st["RepairType"], normalize="index").round(2)
print(f"\n[군집 {K}개 프로필] 단계별 시간은 중앙값(시간)\n", prof.join(mix).to_string())

# 군집별로 '시간이 가장 많이 쓰이는 단계'
stage_share = st.groupby("cluster")[st.columns[:6]].sum()
stage_share = stage_share.div(stage_share.sum(axis=1), axis=0).round(3)
print("\n[군집별 단계 시간 비중]\n", stage_share.to_string())

# ═════════ 2. 개선 효과 시뮬레이션: 점검 대기를 줄이면? ═════════
# 가정: 점검 대기만 줄고 이후 단계 시간은 그대로 (보수적으로 '대기 감소분 = 소요시간 감소분')
s = rep.dropna(subset=["survey_wait_h"]).copy()
s = s[s["survey_wait_h"] >= 0]
base = s["duration_d"]
scen = {"현재": base}
for cap in [72, 48, 24]:
    scen[f"점검 대기 최대 {cap}시간"] = base - (s["survey_wait_h"] - cap).clip(lower=0) / 24
for r in [0.2, 0.3]:
    scen[f"점검 대기 {int(r*100)}% 단축"] = base - s["survey_wait_h"] * r / 24
sim = pd.DataFrame({k: {"소요일_중앙값": v.median(), "소요일_평균": v.mean(), "소요일_p90": v.quantile(0.9),
                        "7일초과_비율": (v > 7).mean()} for k, v in scen.items()}).T
sim["평균_감소율"] = 1 - sim["소요일_평균"] / sim.loc["현재", "소요일_평균"]
sim["7일초과_건수(927건당)"] = (sim["7일초과_비율"] * len(rep)).round(0)
print(f"\n[개선 효과 시뮬레이션] 대상 {len(s)}건\n", sim.round(3).to_string())

# ═════════ 3. A/B 테스트 설계: '점검 지연 알림' ═════════
# 설계: 접수 건을 무작위로 A(기존)/B(알림)에 배정, 수리 방식으로 층화 (방식별 대기시간이 크게 다르므로)
# 1차 지표: 점검 대기시간(시간) → 분포가 치우쳐 Mann-Whitney U 검정, 양측 α=0.05, 목표 검정력 80%
# 가드레일: RepairOK(수리 성공) 비율, 전체 소요시간, 점검 담당자 1인당 처리 건수
w = s[["RepairType", "survey_wait_h"]].reset_index(drop=True)
daily = len(rep) / ((rep["start_time"].max() - rep["start_time"].min()).total_seconds() / 86400)

def power(n_per_arm, effect, adjust, sims=400):
    hit = 0
    for _ in range(sims):
        a = w.sample(n_per_arm, replace=True, random_state=rng.integers(1e9)).copy()
        b = w.sample(n_per_arm, replace=True, random_state=rng.integers(1e9)).copy()
        b["survey_wait_h"] *= (1 - effect)                    # 처리군: 대기시간이 effect만큼 줄었다고 가정
        if not adjust:                                        # 단순 비교: Mann-Whitney U
            p = stats.mannwhitneyu(a["survey_wait_h"], b["survey_wait_h"]).pvalue
        else:                                                 # 수리 방식 보정: 로그 대기시간에서 방식별 평균을 빼고 t검정
            ab = pd.concat([a.assign(g=0), b.assign(g=1)])    # (ANCOVA와 같은 원리, 방식 차이로 생기는 분산 제거)
            y = np.log1p(ab["survey_wait_h"])
            r = y - y.groupby(ab["RepairType"]).transform("mean")
            p = stats.ttest_ind(r[ab["g"] == 0], r[ab["g"] == 1], equal_var=False).pvalue
        hit += p < 0.05
    return hit / sims

grid = [50, 100, 200, 300, 400, 600, 800, 1000]
pw = pd.DataFrame({(m, f"{int(e*100)}% 감소"): {n: power(n, e, m == "방식 보정") for n in grid}
                   for m in ["단순 비교", "방식 보정"] for e in [0.1, 0.2, 0.3]})
pw.index.name = "그룹당 건수"
print(f"\n[A/B 검정력] (현재 하루 약 {daily:.0f}건 접수)\n", pw.to_string())
for c in pw.columns:
    n = next((n for n in grid if pw.loc[n, c] >= 0.8), None)
    print(f"  [{c[0]}] {c[1]} 효과를 80% 검정력으로: " +
          (f"그룹당 {n}건, 총 {2*n}건 ≈ {2*n/daily:.1f}일치 접수" if n else f"그룹당 {grid[-1]}건 초과 필요"))

# ───────── 저장 ─────────
prof.join(mix).to_csv(f"{OUT_DIR}/cluster_profile.csv", encoding="utf-8-sig")
sim.to_csv(f"{OUT_DIR}/improvement_simulation.csv", encoding="utf-8-sig")
pw.columns = [f"{a}_{b}" for a, b in pw.columns]
pw.to_csv(f"{OUT_DIR}/ab_power.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
stage_share.plot.barh(stacked=True, ax=ax[0], colormap="tab20c")
ax[0].set_yticklabels([f"군집 {i} ({prof.loc[i,'cases']}건, 중앙 {prof.loc[i,'소요일_중앙']}일)" for i in stage_share.index])
ax[0].set_title("1. 군집별 단계 시간 비중"); ax[0].legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
keys = ["현재", "점검 대기 최대 72시간", "점검 대기 최대 48시간", "점검 대기 최대 24시간"]
ax[1].bar([k.replace("점검 대기 ", "") for k in keys], sim.loc[keys, "7일초과_비율"] * 100, color=["#9aa5b1", "#8fb3f0", "#5a8fe8", "#2f6fde"])
for i, k in enumerate(keys):
    ax[1].text(i, sim.loc[k, "7일초과_비율"] * 100 + 0.5, f"{sim.loc[k,'7일초과_비율']:.1%}\n평균 -{sim.loc[k,'평균_감소율']:.0%}", ha="center")
ax[1].set_title("2. 점검 대기 상한별 '7일 초과' 비율"); ax[1].set_ylabel("%")
for c in [c for c in pw.columns if c.startswith("단순 비교")]:   # 방식 보정은 결과가 거의 같아 그림에서는 생략
    ax[2].plot(pw.index, pw[c], marker="o", label=c.split("_")[1])
ax[2].axhline(0.8, ls="--", c="gray"); ax[2].set_xlabel("그룹당 건수"); ax[2].set_ylabel("검정력")
ax[2].set_title("3. A/B 테스트 검정력 (점검 대기시간)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_cluster_ab_sim.png", dpi=150)
print(f"\n저장 완료: {OUT_DIR}/cluster_profile.csv, improvement_simulation.csv, ab_power.csv, fig_cluster_ab_sim.png")
