# -*- coding: utf-8 -*-
"""
Q2-3 케이스 소요시간에 영향을 주는 요소 분석 (질문 2-3)
준비: pip3 install scipy   (통계검정에 사용)
      q1~q2_2 실행 후 output/ 에 case_table.pkl, log_clean.pkl, case_rework.csv 가 있어야 함
실행: python3 q2_3.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import os, platform
BASE = os.path.dirname(os.path.abspath(__file__))     # 이 파일이 있는 폴더 기준 (어디서 실행해도 동작)
OUT_DIR = os.path.join(BASE, "output")
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 250, "display.max_columns", 30)

case = pd.read_pickle(f"{OUT_DIR}/case_table.pkl")
log = pd.read_pickle(f"{OUT_DIR}/log_clean.pkl")

# ───────── 0. 분석 대상 만들기 ─────────
# 잘못된 장소로 접수 단계에서 끝난 73건은 수리 과정이 없어 제외 → 수리 진행 케이스 927건
rep = case[~case["wrong_place"]].copy()
Y = "duration_d"                                              # 분석할 결과: 케이스 소요시간(일)

# 접수 시점(로그 시작일로부터 며칠째 접수됐나)
rep["start_day"] = (rep["start_time"] - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400
# 수리 횟수(파트 B에서 계산) → 재작업 여부
rep["n_repairs"] = pd.read_csv(f"{OUT_DIR}/case_rework.csv", index_col=0)["n_repairs"].reindex(rep.index)
rep["rework"] = np.where(rep["n_repairs"] >= 2, "2회 이상", "1회")
# 점검 담당자, 점검 대기시간(고객 안내 완료 → 점검 완료)
sv = log[(log["taskID"] == "Survey") & (log["eventtype"] == "complete")].dropna(subset=["originator"])
rep["survey_by"] = sv.groupby("caseID")["originator"].first().reindex(rep.index).fillna("없음")
done = log[(log["eventtype"] == "complete") & log["taskID"].isin(["InformClientSurvey", "Survey"])]
tmin = done.groupby(["caseID", "taskID"])["timestamp"].min().unstack()
rep["survey_wait_h"] = ((tmin["Survey"] - tmin["InformClientSurvey"]).dt.total_seconds() / 3600).reindex(rep.index)
print("[분석 대상 케이스]", len(rep), "/ 점검 대기시간 계산 가능:", rep["survey_wait_h"].notna().sum())

CAT = {"RepairType": "수리 방식(RepairType)", "EstimatedRepairTime": "예상 수리 시간",
       "RepairCode": "수리 종류(RepairCode)", "route": "수리 경로", "contact": "요청 채널",
       "rework": "재작업(수리 2회 이상)", "survey_by": "점검 담당자"}
NUM = {"start_day": "접수 시점(일)", "start_hour": "접수 시각(시)"}

# ───────── 1. 요소별 그룹 비교: 그룹마다 소요시간이 얼마나 다른가 ─────────
rows = []
for c, name in CAT.items():
    g = rep.groupby(c)[Y].agg(cases="size", median="median", mean="mean",
                              p90=lambda s: s.quantile(0.9)).round(2).reset_index()
    g.insert(0, "factor", name); g = g.rename(columns={c: "group"})
    rows.append(g)
gs = pd.concat(rows)
gs.to_csv(f"{OUT_DIR}/factor_group_stats.csv", index=False, encoding="utf-8-sig")
print("\n[1] 그룹별 소요시간(일) - 주요 요소")
for c in ["RepairType", "contact", "route", "rework"]:
    print(f"\n< {CAT[c]} >")
    print(rep.groupby(c)[Y].agg(cases="size", median="median", mean="mean",
                                p90=lambda s: s.quantile(0.9)).round(2))

# ───────── 2. 통계검정: 그 차이가 우연이 아닌가? ─────────
def kruskal(col):
    d = rep[[col, Y]].dropna()
    groups = [v[Y].values for _, v in d.groupby(col) if len(v) >= 5]
    H, p = stats.kruskal(*groups)
    eps2 = max((H - len(groups) + 1) / (len(d) - len(groups)), 0)   # 효과 크기(0~1)
    return p, eps2

def size_label(e):                                                   # 관례적 기준
    return "큼" if e >= 0.14 else "중간" if e >= 0.06 else "작음" if e >= 0.01 else "거의 없음"

res = []
for c, name in CAT.items():
    p, e = kruskal(c)
    res.append((name, "그룹 비교(Kruskal-Wallis)", p, round(e, 3), size_label(e), "유의" if p < 0.05 else "유의하지 않음"))
for c, name in NUM.items():
    d = rep[[c, Y]].dropna()
    r, p = stats.spearmanr(d[c], d[Y])
    res.append((name, "상관(Spearman)", p, round(abs(r), 3), size_label(r * r), "유의" if p < 0.05 else "유의하지 않음"))
tests = pd.DataFrame(res, columns=["요소", "검정", "p값", "효과크기", "효과", "판정"]).sort_values("효과크기", ascending=False)
tests["p값"] = tests["p값"].apply(lambda v: f"{v:.2e}" if v < 0.001 else f"{v:.3f}")
tests.to_csv(f"{OUT_DIR}/factor_tests.csv", index=False, encoding="utf-8-sig")
print("\n[2] 통계검정 결과 (p값<0.05이면 우연이라 보기 어려움, 효과크기가 클수록 영향이 큼)")
print(tests.to_string(index=False))

# ───────── 3. 요소끼리 얽혀 있는지 확인 ─────────
print("\n[3] 수리 방식(RepairType)과 다른 요소의 관계")
print("\n< 점검 담당자 x 수리 방식 >"); print(pd.crosstab(rep["survey_by"], rep["RepairType"]))
print("\n< 예상 수리 시간 x 수리 방식 >"); print(pd.crosstab(rep["EstimatedRepairTime"], rep["RepairType"]))
print("\n< 수리 경로 x 수리 방식 >"); print(pd.crosstab(rep["route"], rep["RepairType"]))

# ───────── 4. 설명력: 이 요소를 알면 소요시간 차이의 몇 %를 설명하나(R²) ─────────
y = rep[Y].values
# 다중회귀분석 통해 요인들이 소요시간 분산을 얼마나 설명하는지 계산(R^2)
def r2(cat=(), num=()):
    X = [np.ones((len(rep), 1))]
    if cat:
        X.append(pd.get_dummies(rep[list(cat)].astype(str), drop_first=True).astype(float).values)
    if num:
        X.append(rep[list(num)].astype(float).values)
    X = np.hstack(X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()

single = {name: r2(cat=[c]) for c, name in CAT.items()}
single.update({name: r2(num=[c]) for c, name in NUM.items()})
base = r2(cat=["RepairType"])
extra = {name: r2(cat=["RepairType", c]) - base for c, name in CAT.items() if c != "RepairType"}
extra.update({name: r2(cat=["RepairType"], num=[c]) - base for c, name in NUM.items()})
ex = pd.DataFrame({"단독 설명력(R²)": pd.Series(single).round(3),
                   "수리방식에 더해 추가되는 설명력": pd.Series(extra).round(3)}).sort_values("단독 설명력(R²)", ascending=False)
print("\n[4] 설명력 (1.0 = 소요시간 차이를 100% 설명)")
print(ex)
print("수리 방식 + 접수 시점:", round(r2(cat=["RepairType"], num=["start_day"]), 3),
      "/ 모든 요소 합쳐서:", round(r2(cat=["RepairType", "EstimatedRepairTime", "RepairCode", "route", "contact", "rework"],
                                    num=["start_day"]), 3))
ex.to_csv(f"{OUT_DIR}/factor_explained.csv", encoding="utf-8-sig")

# ───────── 5. 수리 방식별로 보면: 점검 대기시간, 접수 시점 ─────────
print("\n[5-1] 수리 방식별 점검 대기시간(시간): 고객 안내 완료 → 점검 완료")
print(rep.groupby("RepairType")["survey_wait_h"].agg(cases="count", median="median", mean="mean",
                                                    p90=lambda s: s.quantile(0.9)).round(1))
print("\n[5-2] 수리 방식별 접수 시점과 소요시간의 상관(Spearman)")
for t, d in rep.groupby("RepairType"):
    r, p = stats.spearmanr(d["start_day"], d[Y])
    print(f"  {t}형 {len(d)}건: 상관 {r:.3f} (p={p:.3g})")
rep["접수시점구간"] = pd.qcut(rep["start_day"], 4, labels=["초반", "중반 전", "중반 후", "후반"])
bt = rep.groupby(["RepairType", "접수시점구간"], observed=True)[Y].median().unstack().round(2)
print("\n[5-3] 접수 시점 구간별 소요시간 중앙값(일)"); print(bt)

rep[["RepairType", "EstimatedRepairTime", "RepairCode", "route", "contact", "rework", "survey_by",
     "start_day", "start_hour", "survey_wait_h", Y]].to_csv(f"{OUT_DIR}/case_factors.csv", encoding="utf-8-sig")

# ───────── 6. 그림 ─────────
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
types = sorted(rep["RepairType"].unique())
ax[0, 0].boxplot([rep.loc[rep["RepairType"] == t, Y] for t in types]); ax[0, 0].set_xticklabels(types)
ax[0, 0].set_title("수리 방식별 케이스 소요시간(일)")
d2 = [rep.loc[(rep["RepairType"] == t) & rep["survey_wait_h"].notna(), "survey_wait_h"] for t in types]
ax[0, 1].boxplot(d2); ax[0, 1].set_xticklabels(types); ax[0, 1].set_title("수리 방식별 점검 대기시간(시간)")
s = ex["단독 설명력(R²)"].sort_values()
ax[1, 0].barh(s.index, s.values, color="#4472C4"); ax[1, 0].set_title("요소별 단독 설명력(R²)")
for i, v in enumerate(s.values):
    ax[1, 0].text(v, i, f" {v:.2f}", va="center", fontsize=9)
for t in types:
    ax[1, 1].plot(bt.columns.astype(str), bt.loc[t].values, marker="o", label=f"{t}형")
ax[1, 1].set_title("접수 시점 구간별 소요시간 중앙값(일)"); ax[1, 1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_factors.png", dpi=150)
print(f"\n저장 완료: {OUT_DIR}/factor_tests.csv, factor_group_stats.csv, factor_explained.csv, case_factors.csv, fig_factors.png")
