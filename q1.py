# -*- coding: utf-8 -*-
"""
집 수리 프로세스 로그 - Q1 전처리 (repair2.csv 구조 반영)
실행: python3 q1.py
repair2.csv 를 두고 실행하면 output/ 에 결과가 저장됨
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------0. 설정----------
FILE_PATH = "repair2.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ───────── 1. 로드 및 기본 확인 ─────────
df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()
print("shape:", df.shape)
print(df.dtypes)
print(df.head())

print("\n[결측치]")
print(df.isna().sum())
print("\n[중복 행]", df.duplicated().sum())
print("\n[컬럼별 고유값 수]")
print(df.nunique())

cat_cols = [c for c in df.columns if df[c].nunique() <= 20]
print("\n[범주형으로 볼 컬럼]", cat_cols)
for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False).head(15))

# 참고: 이 로그는 속성이 '특정 업무 행'에만 채워져 있음 (예: contact는 FirstContact에만)
print("\n[속성이 채워진 업무]")
for c in ["contact", "RepairType", "objectKey", "RepairInternally",
          "EstimatedRepairTime", "RepairCode", "RepairOK"]:
    print(c, df[df[c].notna()]["taskID"].unique().tolist())

# ───────── 2. 결측 처리 및 타임스탬프 정제 ─────────
# 불리언 컬럼 정리
for c in ["RepairInternally", "RepairOK"]:
    df[c] = df[c].astype("boolean")

# 핵심 컬럼(taskID/eventtype/originator) 결측 행 확인
core_na = df[["taskID", "originator", "eventtype"]].isna().any(axis=1)
print("\n[핵심 컬럼 결측 행]", core_na.sum())

# Date+Time → 원본 timestamp
df["ts_raw"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                              errors="coerce")
print("[date/time 결측으로 변환 불가]", df["ts_raw"].isna().sum())

# 연도 분포 확인: 대부분이 한 연도(1970)이고 일부만 2020~2022/1905인지 확인
print("\n[연도별 행 수]")
print(df["ts_raw"].dt.year.value_counts().sort_index())

# 가정: 가장 많은 연도(1970년 1월 구간)가 정상 기록, 그 외 연도는 이상 타임스탬프로 간주
main_year = df["ts_raw"].dt.year.mode()[0]
df["ts_abnormal"] = df["ts_raw"].notna() & (df["ts_raw"].dt.year != main_year)
df["timestamp"] = df["ts_raw"].where(~df["ts_abnormal"])   # 이상값은 NaT 처리
print(f"[기준 연도] {main_year}  / 이상 타임스탬프 {df['ts_abnormal'].sum()}행 "
      f"({df['ts_abnormal'].mean():.1%}) → NaT 처리")
print("[유효 타임스탬프 범위]", df["timestamp"].min(), "~", df["timestamp"].max())

# 정렬: 케이스 → 시각 순 (시각이 없는 행은 맨 뒤)
df = df.sort_values(["caseID", "timestamp"], na_position="last").reset_index(drop=True)

# ───────── 3. 케이스 단위 파생변수 ─────────
g = df.groupby("caseID")
case = pd.DataFrame({
    "start_time": g["timestamp"].min(),
    "end_time": g["timestamp"].max(),
    "n_events": g.size(),
    "n_tasks": g["taskID"].nunique(),
    "n_resources": g["originator"].nunique(),
    "n_bad_ts": g["timestamp"].apply(lambda s: s.isna().sum()),
})
# 1) 소요시간 관련 파생변수
case["duration_h"] = (case["end_time"] - case["start_time"]).dt.total_seconds() / 3600
case["duration_d"] = case["duration_h"] / 24

# 속성은 특정 업무 행에만 있고 케이스 안에서 값이 바뀌지 않음(확인됨) → first()로 대표
attr_cols = ["contact", "RepairType", "objectKey", "RepairInternally",
             "EstimatedRepairTime", "RepairCode", "RepairOK"]
for c in attr_cols:
    print(c, "케이스 내 값이 2개 이상인 케이스 수:", (g[c].nunique() > 1).sum())
case = case.join(g[attr_cols].first())

# 경로 구분 플래그 (3일차 영향 요인 분석에서 사용)
tasks_by_case = g["taskID"].apply(set)
# 2) 수리 경로 및 예외 처리 관련 파생변수
case["wrong_place"] = tasks_by_case.apply(lambda s: "InformClientWrongPlace" in s)
case["route"] = tasks_by_case.apply(
    lambda s: "InternRepair" if "InternRepair" in s else
              "ExternRepair" if "ExternRepair" in s else
              "ImmediateRepair" if "ImmediateRepair" in s else "none")

# 3) 시작 시점(시간/요일) 파생변수
case["start_hour"] = case["start_time"].dt.hour
case["start_weekday"] = case["start_time"].dt.dayofweek   # 가정: 1970-01-01 기준 가상 달력

print("\n[케이스 수]", len(case), "/ 소요시간 계산 가능:", case["duration_d"].notna().sum())
print("[케이스 소요시간(일) 요약]")
print(case["duration_d"].describe())
print("\n[경로별 케이스 수]\n", case["route"].value_counts())
print("[잘못된 장소(WrongPlace) 케이스]", case["wrong_place"].sum())

# ───────── 4. start/complete 짝짓기 → 업무 단위 테이블 ─────────
# start가 존재하는 업무만 짝짓기 대상 (FirstContact, RepairReady 등은 complete만 있는 시점 이벤트)
paired_tasks = df.loc[df["eventtype"] == "start", "taskID"].unique()
starts = df[(df["eventtype"] == "start") & df["taskID"].isin(paired_tasks)].copy()
ends = df[(df["eventtype"] == "complete") & df["taskID"].isin(paired_tasks)].copy()
# 같은 업무가 여러 번 반복될 수 있으니 순번 부여
for d in (starts, ends):
    # 1) 순회 및 짝짓기용 변수
    d["seq"] = d.groupby(["caseID", "taskID", "originator"]).cumcount()

tasks = starts.merge(
    ends[["caseID", "taskID", "originator", "seq", "timestamp"]],
    on=["caseID", "taskID", "originator", "seq"],
    how="outer", suffixes=("_start", "_end"),
)
# 2) 실제 작업 소요 시간 파생변수
tasks["work_min"] = (tasks["timestamp_end"] - tasks["timestamp_start"]).dt.total_seconds() / 60

print("\n[짝짓기 대상 업무]", sorted(paired_tasks))
unmatched = tasks[tasks["timestamp_start"].isna() | tasks["timestamp_end"].isna()]
print("[짝 안 맞거나 시각 결측인 업무 수, 업무별]")
print(unmatched["taskID"].value_counts())
print("[work_min 음수]", (tasks["work_min"] < 0).sum())
print(tasks["work_min"].describe())

# 대기시간: 같은 케이스에서 이전 업무 complete → 다음 업무 start
tasks = tasks.sort_values(["caseID", "timestamp_start"])
tasks["prev_end"] = tasks.groupby("caseID")["timestamp_end"].shift()
tasks["wait_min"] = (tasks["timestamp_start"] - tasks["prev_end"]).dt.total_seconds() / 60

# ───────── 5. 기초 시각화 ─────────
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
case["duration_d"].hist(bins=40, ax=ax[0, 0]); ax[0, 0].set_title("케이스 소요시간 분포(일)")
case["n_events"].hist(bins=range(1, 21), ax=ax[0, 1]); ax[0, 1].set_title("케이스당 이벤트 수")
df["taskID"].value_counts().plot.barh(ax=ax[0, 2]); ax[0, 2].set_title("업무별 이벤트 수")
case["contact"].value_counts().plot.bar(ax=ax[1, 0]); ax[1, 0].set_title("요청 채널별 케이스 수")
case["RepairType"].value_counts().plot.bar(ax=ax[1, 1]); ax[1, 1].set_title("수리 방식별 케이스 수")
df["originator"].value_counts().plot.bar(ax=ax[1, 2]); ax[1, 2].set_title("작업자별 이벤트 수")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_basic.png", dpi=150)

# ───────── 6. 저장 ─────────
df.to_pickle(f"{OUT_DIR}/log_clean.pkl")
case.to_pickle(f"{OUT_DIR}/case_table.pkl")
tasks.to_pickle(f"{OUT_DIR}/task_table.pkl")
