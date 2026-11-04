# -*- coding: utf-8 -*-
"""
Q2-1 현재 프로세스: 프로세스 맵(DFG) + 경로(variant) 분석
준비: q1s.py 실행 후 output/log_clean.pkl, case_table.pkl 이 있어야 함
실행: python3 q2_1.py   (추가 설치 없음: pandas, numpy, matplotlib만 사용)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import os, platform
BASE = os.path.dirname(os.path.abspath(__file__))     # 이 파일이 있는 폴더 기준 (어디서 실행해도 동작)
OUT_DIR = os.path.join(BASE, "output")
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False
MIN_EDGE = 5          # 맵에는 이 건수 이상인 연결만 그림 (전체는 CSV에 저장)

log = pd.read_pickle(f"{OUT_DIR}/log_clean.pkl")
case = pd.read_pickle(f"{OUT_DIR}/case_table.pkl")

# ───────── 1. 활동(activity) 이벤트 만들기 ─────────
# 업무 하나를 이벤트 하나로 본다: complete 기록을 쓰고, complete가 없는 ExternRepair는 start를 쓴다.
act = log[(log["eventtype"] == "complete") |
          ((log["eventtype"] == "start") & (log["taskID"] == "ExternRepair"))].copy()
act = act[act["taskID"].notna()]          # 업무명이 비어 있는 행(2건)은 제외

# 업무 진행 단계 (Q1에서 데이터로 확인한 표준 순서). 이상 시각 이벤트의 위치 복원에 사용.
ORDER = ["FirstContact", "InformClientWrongPlace", "MakeTicket", "ArrangeSurvey",
         "InformClientSurvey", "Survey", "ImmediateRepair", "InternRepair", "ExternRepair",
         "RepairReady", "ReadyInformClient", "TicketReady", "SendTicketToFinAdmin"]
rank = {t: i for i, t in enumerate(ORDER)} # 업무마다번호
act["rank"] = act["taskID"].map(rank)
CLOSING = {"ReadyInformClient", "TicketReady", "SendTicketToFinAdmin"}   # 선후가 일정치 않은 마무리 3종

print("[활동 이벤트]", len(act), "/ 이상 시각으로 비어 있는 것:", act["timestamp"].isna().sum())
print("[이상 시각 활동이 있는 케이스]", act.groupby("caseID")["timestamp"].apply(lambda s: s.isna().any()).sum(), "/ 1000")


def build_sequence(ev):
    """케이스 하나의 업무 순서를 만든다.
    1) 시각이 있는 이벤트는 시각 순(같으면 단계 순)으로 정렬
    2) 시각이 없는 이벤트는 '단계 순서상 직전 단계 이벤트 바로 뒤'에 끼워 넣는다 (가정)
    3) 마무리 3종은 하나의 'Closing'으로 합친다 (3개의 선후가 일정치 않아서)
    """
    valid = ev[ev["timestamp"].notna()].sort_values(["timestamp", "rank"])
    missing = ev[ev["timestamp"].isna()].sort_values("rank")
    out = list(zip(valid["taskID"], valid["rank"]))
    for t, r in zip(missing["taskID"], missing["rank"]):
        pos = 0
        for i, (_, rr) in enumerate(out):
            if rr <= r:
                pos = i + 1
        out.insert(pos, (t, r))
    seq, closing_added = [], False
    for t, _ in out:
        if t in CLOSING:
            if not closing_added:
                seq.append("Closing"); closing_added = True
        else:
            seq.append(t); closing_added = False
    return tuple(seq)

seqs = pd.Series({cid: build_sequence(ev) for cid, ev in act.groupby("caseID")})
seqs.index.name = "caseID"
print("[업무 순서를 만든 케이스]", len(seqs))

# ───────── 2. 경로(variant) 분석 ─────────
var = seqs.rename("path").to_frame().join(case["duration_d"])
vt = (var.groupby("path")
         .agg(cases=("duration_d", "size"), median_days=("duration_d", "median"))
         .sort_values("cases", ascending=False).reset_index())
# 비율
vt["pct"] = (vt["cases"] / vt["cases"].sum() * 100).round(1)
# 누적 비율
vt["cum_pct"] = vt["pct"].cumsum().round(1)
vt["variant"] = [f"V{i+1}" for i in range(len(vt))]
vt["path_str"] = vt["path"].apply(lambda p: " → ".join(p))
vt["n_steps"] = vt["path"].apply(len)
vt[["variant", "cases", "pct", "cum_pct", "median_days", "n_steps", "path_str"]].to_csv(
    f"{OUT_DIR}/variants.csv", index=False, encoding="utf-8-sig")
path_to_var = dict(zip(vt["path"], vt["variant"]))
case["variant"] = [path_to_var.get(seqs.get(c)) for c in case.index]
case[["variant"]].to_csv(f"{OUT_DIR}/case_variant.csv", encoding="utf-8-sig")

print(f"\n[경로(variant) 수] {len(vt)}개")
print("[상위 10개 경로]")
pd.set_option("display.width", 250, "display.max_colwidth", 140)
print(vt[["variant", "cases", "pct", "cum_pct", "median_days", "path_str"]].head(10).to_string(index=False))

# ───────── 3. 프로세스 맵(DFG): 업무 → 다음 업무 연결 건수 ─────────
rows = []
for cid, s in seqs.items():
    path = ("Start",) + s + ("End",)
    rows += [(a, b, cid) for a, b in zip(path[:-1], path[1:])]
ed = pd.DataFrame(rows, columns=["from", "to", "caseID"])
dfg = (ed.groupby(["from", "to"])["caseID"].agg(count="size", cases="nunique")
         .reset_index().sort_values("count", ascending=False))
dfg.to_csv(f"{OUT_DIR}/dfg_edges.csv", index=False, encoding="utf-8-sig")
print("\n[연결(edge) 수]", len(dfg), "/ 상위 15개")
print(dfg.head(15).to_string(index=False))
loops = dfg[dfg["from"] == dfg["to"]]
print("\n[같은 업무가 연속 반복된 연결(루프)]")
print(loops.to_string(index=False) if len(loops) else "없음")

# 맵 그리기 (노드 위치는 직접 지정한 왼쪽→오른쪽 배치)
pos = {"Start": (0, 0), "FirstContact": (1, 0), "InformClientWrongPlace": (2, -2.2),
       "MakeTicket": (2, 0), "ArrangeSurvey": (3, 0), "InformClientSurvey": (4, 0),
       "Survey": (5, 0), "ImmediateRepair": (6.3, 1.3), "InternRepair": (6.3, 0),
       "ExternRepair": (6.3, -1.3), "RepairReady": (7.6, 0), "Closing": (8.8, 0), "End": (10, 0)}
fig, ax = plt.subplots(figsize=(20, 8))
maxc = dfg["count"].max()
shown = dfg[(dfg["count"] >= MIN_EDGE) & (dfg["from"] != dfg["to"])]
for _, r in shown.iterrows():
    (x1, y1), (x2, y2) = pos[r["from"]], pos[r["to"]]
    same_col = abs(x1 - x2) < 0.5
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                          lw=0.8 + 6 * r["count"] / maxc, color="#4472C4", alpha=0.75,
                          shrinkA=38, shrinkB=38,
                          connectionstyle="arc3,rad=0.35" if same_col else "arc3,rad=0.0")
    ax.add_patch(arr)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + (0.35 if same_col else 0.18)
    ax.text(mx, my, str(r["count"]), ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
loop_n = loops.set_index("from")["count"].to_dict()
for n, (x, y) in pos.items():
    label = n + (f"\n↻ 반복 {loop_n[n]}건" if n in loop_n else "")
    ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFF2CC" if n not in ("Start", "End") else "#D9D9D9",
                      ec="#7F7F7F"))
ax.set_xlim(-0.8, 10.8); ax.set_ylim(-3.2, 2.5); ax.axis("off")
ax.set_title(f"프로세스 맵 (업무 → 다음 업무, {MIN_EDGE}건 이상, 화살표 굵기=건수)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dfg_map.png", dpi=150)
print(f"\n저장 완료: {OUT_DIR}/variants.csv, dfg_edges.csv, case_variant.csv, dfg_map.png")
