# -*- coding: utf-8 -*-
"""
Q2-2 문제점: 업무별 작업시간 + 단계 간 소요시간(병목) + 작업자 부하 + 재작업 비교
준비: q1.py 실행 후 output/log_clean.pkl, case_table.pkl, task_table.pkl 이 있어야 함
실행: python3 q2_2.py   (추가 설치 없음)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os, platform
BASE = os.path.dirname(os.path.abspath(__file__))     # 이 파일이 있는 폴더 기준 (어디서 실행해도 동작)
OUT_DIR = os.path.join(BASE, "output")
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 250, "display.max_columns", 30)

log = pd.read_pickle(f"{OUT_DIR}/log_clean.pkl")
case = pd.read_pickle(f"{OUT_DIR}/case_table.pkl")
tasks = pd.read_pickle(f"{OUT_DIR}/task_table.pkl")

# ───────── 1. 업무별 작업시간 (시작 → 완료) ─────────
# task_table의 work_min 사용. 시각이 비어 있거나 음수인 업무는 제외.
w = tasks.dropna(subset=["work_min"])
print("[작업시간 계산 가능한 업무]", len(w), "/ 음수 제외:", (w["work_min"] < 0).sum())
w = w[w["work_min"] >= 0]
work = (w.groupby("taskID")["work_min"]
          .agg(n="size", median="median", mean="mean",
               p90=lambda s: s.quantile(0.9), max="max").round(1))
print("\n[업무별 작업시간(분)] ※ ExternRepair는 완료 기록이 없어 제외")
print(work)
work.to_csv(f"{OUT_DIR}/work_time_by_task.csv", encoding="utf-8-sig")

# ───────── 2. 단계 간 소요시간 (병목 찾기) ─────────
# 파트 A와 같은 방식으로 케이스별 업무 순서를 만들되, 이번에는 각 업무의 완료 시각도 함께 들고 간다.
act = log[(log["eventtype"] == "complete") |
          ((log["eventtype"] == "start") & (log["taskID"] == "ExternRepair"))].copy()
act = act[act["taskID"].notna()]
ORDER = ["FirstContact", "InformClientWrongPlace", "MakeTicket", "ArrangeSurvey",
         "InformClientSurvey", "Survey", "ImmediateRepair", "InternRepair", "ExternRepair",
         "RepairReady", "ReadyInformClient", "TicketReady", "SendTicketToFinAdmin"]
rank = {t: i for i, t in enumerate(ORDER)}
act["rank"] = act["taskID"].map(rank)
CLOSING = {"ReadyInformClient", "TicketReady", "SendTicketToFinAdmin"}


def build_sequence_with_time(ev):
    """[(업무, 완료시각), ...] 를 순서대로 돌려준다. 시각이 비어 있던 업무는 시각이 NaT."""
    valid = ev[ev["timestamp"].notna()].sort_values(["timestamp", "rank"])
    missing = ev[ev["timestamp"].isna()].sort_values("rank")
    out = list(zip(valid["taskID"], valid["rank"], valid["timestamp"]))
    for t, r in zip(missing["taskID"], missing["rank"]):          # 파트 A와 같은 복원 규칙
        pos = 0
        for i, (_, rr, _) in enumerate(out):
            if rr <= r:
                pos = i + 1
        out.insert(pos, (t, r, pd.NaT))
    res, closing = [], []
    for t, _, ts in out:
        if t in CLOSING:
            closing.append(ts)
        else:
            if closing:
                res.append(("Closing", closing)); closing = []
            res.append((t, [ts]))
    if closing:
        res.append(("Closing", closing))
    # Closing은 3개 중 하나라도 시각이 없으면 시각 불명으로 둔다 (가장 이른 시각을 대표값으로 사용)
    return [(t, min(x) if all(pd.notna(i) for i in x) else pd.NaT) for t, x in res]


rows = []
for cid, ev in act.groupby("caseID"):
    s = build_sequence_with_time(ev)
    for (a, ta), (b, tb) in zip(s[:-1], s[1:]):                   # 바로 이어지는 두 업무
        rows.append((cid, a, b, ta, tb))
tr = pd.DataFrame(rows, columns=["caseID", "A", "B", "tA", "tB"])
tr["hours"] = (tr["tB"] - tr["tA"]).dt.total_seconds() / 3600
# 두 업무 모두 시각이 있고 바로 이어진 경우만 사용 (사이에 시각 없는 업무가 끼면 시간이 부풀려지므로)
ok = tr.dropna(subset=["hours"])
print(f"\n[단계 간 전이] 전체 {len(tr)}건 중 시각이 모두 있는 {len(ok)}건 사용 / 음수: {(ok['hours'] < 0).sum()}")

# 병목
tt = (ok.groupby(["A", "B"])["hours"]
        .agg(n="size", median_h="median", mean_h="mean",
             p90_h=lambda s: s.quantile(0.9), total_h="sum").reset_index())

# 모든 단계의 시간을 다 더했을 때 이 단계가 차지하는 비율
tt["share_pct"] = (tt["total_h"] / tt["total_h"].sum() * 100).round(1)
# 대략적인 대기시간 = 단계 소요 중앙값 - 다음 업무의 작업시간 중앙값 (추정치)
tt["work_h"] = tt["B"].map(work["median"] / 60)
tt["wait_est_h"] = (tt["median_h"] - tt["work_h"]).clip(lower=0)
tt = tt.round(2).sort_values("total_h", ascending=False)
tt["step"] = tt["A"] + " → " + tt["B"]
tt.to_csv(f"{OUT_DIR}/transition_time.csv", index=False, encoding="utf-8-sig")
print("\n[단계 간 소요시간(시간): 이전 업무 완료 → 다음 업무 완료] 누적시간 큰 순")
print(tt[["step", "n", "median_h", "mean_h", "p90_h", "share_pct", "wait_est_h"]].head(10).to_string(index=False))

# 병목 그림: 평균 소요시간과 전체 시간 점유율 (표본 30건 이상, 거의 0시간인 단계는 제외)
big = tt[(tt["n"] >= 30) & (tt["mean_h"] > 0.5)].sort_values("mean_h")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].barh(big["step"], big["mean_h"], color="#4472C4")
ax[0].set_title("단계별 평균 소요시간 (시간)")
for y, (m, med) in enumerate(zip(big["mean_h"], big["median_h"])):
    ax[0].text(m, y, f" 평균 {m:.1f} / 중앙값 {med:.1f}", va="center", fontsize=9)
ax[0].set_xlim(0, big["mean_h"].max() * 1.5)
b2 = big.sort_values("share_pct")
ax[1].barh(b2["step"], b2["share_pct"], color="#C55A11")
ax[1].set_title("전체 단계 소요시간 중 차지하는 비율 (%)")
for y, v in enumerate(b2["share_pct"]):
    ax[1].text(v, y, f" {v}%", va="center", fontsize=9)
ax[1].set_xlim(0, b2["share_pct"].max() * 1.2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_bottleneck.png", dpi=150)

# ───────── 3. 작업자 부하 ─────────
ev_cnt = log["originator"].value_counts()
load = (w.groupby("originator")["work_min"]
          .agg(n="size", total_h=lambda s: s.sum() / 60, median_min="median").round(1)
          .sort_values("total_h", ascending=False))
load["event_share_pct"] = (ev_cnt / ev_cnt.sum() * 100).round(1).reindex(load.index)
print("\n[작업자별 작업시간 합계(시작~완료 업무만)]")
print(load.head(10))
load.to_csv(f"{OUT_DIR}/originator_load.csv", encoding="utf-8-sig")
print("\n[전체 이벤트 중 상위 3 작업자 비중]", round(ev_cnt.head(3).sum() / ev_cnt.sum() * 100, 1), "%")

# ───────── 4. 재작업(수리 2회 이상) 비교 ─────────
REPAIR = {"ImmediateRepair", "InternRepair", "ExternRepair"}
n_rep = {}
for cid, ev in act.groupby("caseID"):
    n_rep[cid] = sum(1 for t, _ in build_sequence_with_time(ev) if t in REPAIR)
case["n_repairs"] = pd.Series(n_rep)
rep_case = case[~case["wrong_place"]]
cmp_ = (rep_case.groupby(rep_case["n_repairs"].clip(upper=2))["duration_d"]
                .agg(cases="size", median="median", mean="mean",
                     p90=lambda s: s.quantile(0.9)).round(2))
cmp_.index = ["수리 1회", "수리 2회 이상"] if len(cmp_) == 2 else cmp_.index
print("\n[수리 횟수별 케이스 소요시간(일)]")
print(cmp_)
case[["n_repairs"]].to_csv(f"{OUT_DIR}/case_rework.csv", encoding="utf-8-sig")
print(f"\n저장 완료: {OUT_DIR}/work_time_by_task.csv, transition_time.csv, originator_load.csv, case_rework.csv, fig_bottleneck.png")
