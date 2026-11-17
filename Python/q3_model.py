# -*- coding: utf-8 -*-
"""
Q3 소요시간 예측 모델 (분류 + 회귀) - scikit-learn
준비: pip3 install scikit-learn
      q1~q2_3 실행 후 output/ 에 case_table.pkl, log_clean.pkl 이 있어야 함
실행: python3 q3_model.py

[질문] 요청이 '오래 걸릴지'를 미리 알 수 있다면, 점검 일정을 먼저 잡아 대기를 줄일 수 있다.
  A. 접수 직후   : 이 요청이 7일을 넘길까?           → 분류 (Logistic / RandomForest / HistGB)
  B. 점검 완료 후 : 완료까지 며칠이 더 남았나?       → 회귀 (Ridge / RandomForest / HistGB)

[누수 방지] 예측 시점에 아직 모르는 정보는 변수에서 뺀다.
  - 접수 직후에는 점검 결과(예상 수리시간, 수리 코드, 내부 수리 여부), 점검 담당자, 실제 경로를 모름
  - 경로(route)·재작업 여부·RepairOK는 모두 끝나야 알 수 있으므로 두 모델 모두 제외

[검증] 케이스를 접수 시각 순으로 정렬해 앞 80% = 학습, 뒤 20% = 테스트
  - 하이퍼파라미터는 학습 구간 안에서만 TimeSeriesSplit(5) 교차검증으로 고름
  - 테스트 구간은 마지막에 한 번만 사용 (말하는 일기장 때 테스트셋으로 epoch를 고른 한계를 보완)
"""
import os, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, \
    recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
plt.rcParams["font.family"] = {"Darwin": "AppleGothic", "Windows": "Malgun Gothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False
pd.set_option("display.width", 250, "display.max_columns", 30)
SEED = 42
LONG_DAYS = 7                                                  # '오래 걸림' 기준: 7일 초과 (약 상위 25%)

case = pd.read_pickle(f"{OUT_DIR}/case_table.pkl")
log = pd.read_pickle(f"{OUT_DIR}/log_clean.pkl")

# ───────── 0. 분석 대상: 수리가 진행된 927건 (잘못된 장소 73건 제외, Q2-3과 동일) ─────────
rep = case[~case["wrong_place"]].copy().sort_values("start_time")

# ───────── 1. 변수 만들기 ─────────
# (1) 접수 시점에 알 수 있는 업무 부하: 접수 순간 아직 안 끝난 케이스 수 / 점검을 기다리는 케이스 수
done = log[log["eventtype"] == "complete"]
first = done.groupby(["caseID", "taskID"])["timestamp"].min().unstack()
inform_t, survey_t = first["InformClientSurvey"], first["Survey"]
all_start, all_end = case["start_time"].values, case["end_time"].values

def open_at(t):
    return int(((all_start < t) & (all_end > t)).sum())

def wait_survey_at(t):
    m = (inform_t < t) & ((survey_t > t) | survey_t.isna())
    return int(m.sum())

rep["open_cases"] = [open_at(t) for t in rep["start_time"].values]
rep["survey_queue"] = [wait_survey_at(t) for t in rep["start_time"]]

# (2) 점검 완료 시점에 알 수 있는 정보
rep["survey_done"] = survey_t.reindex(rep.index)
rep["survey_wait_h"] = ((survey_t - inform_t).dt.total_seconds() / 3600).reindex(rep.index)
sv = log[(log["taskID"] == "Survey") & (log["eventtype"] == "complete")].dropna(subset=["originator"])
rep["survey_by"] = sv.groupby("caseID")["originator"].first().reindex(rep.index).fillna("없음")
rep["elapsed_d"] = (rep["survey_done"] - rep["start_time"]).dt.total_seconds() / 86400
rep["remain_d"] = (rep["end_time"] - rep["survey_done"]).dt.total_seconds() / 86400
rep["RepairCode"] = rep["RepairCode"].astype("Int64").astype(str)
rep["RepairInternally"] = rep["RepairInternally"].astype(str)
rep["long"] = (rep["duration_d"] > LONG_DAYS).astype(int)

print(f"[분석 대상] {len(rep)}건 / 7일 초과 {rep['long'].sum()}건 ({rep['long'].mean():.1%})")

# ───────── 2. 시간 순 분할 (앞 80% 학습, 뒤 20% 테스트) ─────────
cut = int(len(rep) * 0.8)
train, test = rep.iloc[:cut], rep.iloc[cut:]
print(f"[분할] 학습 {len(train)}건 ({train['start_time'].min():%m-%d}~{train['start_time'].max():%m-%d}) / "
      f"테스트 {len(test)}건 ({test['start_time'].min():%m-%d}~{test['start_time'].max():%m-%d})")
cv = TimeSeriesSplit(n_splits=5)


def prep(cat, num):
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler(), num)])


# ═════════ A. 분류: 접수 직후, 7일을 넘길까? ═════════
A_CAT = ["RepairType", "contact"]
A_NUM = ["start_hour", "start_weekday", "open_cases", "survey_queue"]
XA_tr, XA_te = train[A_CAT + A_NUM], test[A_CAT + A_NUM]
yA_tr, yA_te = train["long"], test["long"]

clf_space = {
    "로지스틱 회귀": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                 {"m__C": [0.01, 0.1, 1, 10]}),
    "랜덤 포레스트": (RandomForestClassifier(random_state=SEED, class_weight="balanced"),
                 {"m__n_estimators": [200, 500], "m__max_depth": [3, 5, 8, None],
                  "m__min_samples_leaf": [1, 5, 20]}),
    "그래디언트 부스팅": (HistGradientBoostingClassifier(random_state=SEED),
                   {"m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3, 5],
                    "m__max_iter": [100, 300], "m__min_samples_leaf": [10, 30]}),
}
resA, fittedA = [], {}
dummy = DummyClassifier(strategy="prior").fit(XA_tr, yA_tr)
p0 = dummy.predict_proba(XA_te)[:, 1]
resA.append({"model": "기준선(무작위)", "best_params": "-", "cv_auc": np.nan,
             "test_auc": 0.5, "test_pr_auc": round(yA_te.mean(), 3)})
# 기준선 2: '수리 방식별 7일 초과 비율'만 쓰는 단순 규칙 → 모델이 이 규칙보다 나은지가 진짜 기준
rule = XA_te["RepairType"].map(train.groupby("RepairType")["long"].mean()).fillna(yA_tr.mean())
resA.append({"model": "기준선(수리방식 규칙)", "best_params": "-", "cv_auc": np.nan,
             "test_auc": round(roc_auc_score(yA_te, rule), 3),
             "test_pr_auc": round(average_precision_score(yA_te, rule), 3)})
for name, (est, grid) in clf_space.items():
    gs = GridSearchCV(Pipeline([("p", prep(A_CAT, A_NUM)), ("m", est)]), grid,
                      cv=cv, scoring="roc_auc", n_jobs=-1).fit(XA_tr, yA_tr)
    p = gs.predict_proba(XA_te)[:, 1]
    fittedA[name] = (gs, p)
    resA.append({"model": name, "best_params": {k[3:]: v for k, v in gs.best_params_.items()},
                 "cv_auc": round(gs.best_score_, 3), "test_auc": round(roc_auc_score(yA_te, p), 3),
                 "test_pr_auc": round(average_precision_score(yA_te, p), 3)})
resA = pd.DataFrame(resA)
print("\n[A] 접수 직후 '7일 초과' 분류 (튜닝은 학습 구간 CV, 점수는 테스트 구간)")
print(resA.to_string(index=False))

# 교차검증 점수가 가장 좋은 모델을 고른다 (테스트 점수로 고르지 않음)
bestA = resA.dropna(subset=["cv_auc"]).sort_values("cv_auc", ascending=False).iloc[0]["model"]
gsA, pA = fittedA[bestA]
# 운영 기준: 점검 일정을 우선 잡을 수 있는 용량을 '접수 건의 30%'로 가정 → 학습 구간 확률 상위 30% 기준선
thr = np.quantile(gsA.predict_proba(XA_tr)[:, 1], 0.70)
flag = (pA >= thr).astype(int)
prec, rec = precision_score(yA_te, flag), recall_score(yA_te, flag)
print(f"\n[A] 선택 모델: {bestA} / 확률 상위 30%를 '우선 점검'으로 지정하면")
print(f"    테스트 구간 지정 {flag.sum()}건 중 실제 7일 초과 {int((flag & yA_te.values).sum())}건 "
      f"(정밀도 {prec:.1%}, 무작위 지정 시 {yA_te.mean():.1%}) / 7일 초과 건의 {rec:.1%}를 미리 찾음")

# 같은 수리 방식 안에서도 오래 걸릴 건을 구분하는가? (규칙은 같은 방식이면 점수가 같아 구분 불가)
for t in ["E", "P"]:
    m = (XA_te["RepairType"] == t).values
    if yA_te[m].nunique() == 2:
        print(f"    {t}형 내부 구분력 AUC {roc_auc_score(yA_te[m], pA[m]):.3f} "
              f"({m.sum()}건 중 7일 초과 {int(yA_te[m].sum())}건, 규칙은 0.5)")

impA = permutation_importance(gsA, XA_te, yA_te, scoring="roc_auc", n_repeats=30, random_state=SEED)
impA = pd.Series(impA.importances_mean, index=A_CAT + A_NUM).sort_values(ascending=False).round(3)
print("\n[A] 변수 중요도 (테스트 구간 permutation, AUC 감소량)\n", impA.to_string())

# ═════════ B. 회귀: 점검 완료 직후, 완료까지 며칠 남았나? ═════════
B_CAT = ["RepairType", "contact", "RepairCode", "RepairInternally", "survey_by"]
B_NUM = ["EstimatedRepairTime", "survey_wait_h", "elapsed_d", "open_cases", "start_hour"]
trB, teB = train.dropna(subset=["remain_d", "survey_wait_h"]), test.dropna(subset=["remain_d", "survey_wait_h"])
XB_tr, XB_te, yB_tr, yB_te = trB[B_CAT + B_NUM], teB[B_CAT + B_NUM], trB["remain_d"], teB["remain_d"]

# 기준선: 학습 구간의 '수리 방식별 남은 일수 중앙값' (지금 현장에서 쓸 수 있는 가장 단순한 규칙)
med = yB_tr.groupby(trB["RepairType"]).median()
base_pred = teB["RepairType"].map(med).fillna(yB_tr.median())
resB = [{"model": "기준선(수리방식별 중앙값)", "best_params": "-", "cv_mae": np.nan,
         "test_mae": round(mean_absolute_error(yB_te, base_pred), 2),
         "test_r2": round(r2_score(yB_te, base_pred), 3)}]
reg_space = {
    "릿지 회귀": (Ridge(), {"m__alpha": [0.1, 1, 10, 100]}),
    "랜덤 포레스트": (RandomForestRegressor(random_state=SEED),
                 {"m__n_estimators": [300], "m__max_depth": [4, 8, None], "m__min_samples_leaf": [1, 5, 20]}),
    "그래디언트 부스팅": (HistGradientBoostingRegressor(random_state=SEED, loss="absolute_error"),
                   {"m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 3, 5],
                    "m__max_iter": [100, 300], "m__min_samples_leaf": [10, 30]}),
}
fittedB = {}
for name, (est, grid) in reg_space.items():
    gs = GridSearchCV(Pipeline([("p", prep(B_CAT, B_NUM)), ("m", est)]), grid,
                      cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(XB_tr, yB_tr)
    pr = gs.predict(XB_te)
    fittedB[name] = (gs, pr)
    resB.append({"model": name, "best_params": {k[3:]: v for k, v in gs.best_params_.items()},
                 "cv_mae": round(-gs.best_score_, 2), "test_mae": round(mean_absolute_error(yB_te, pr), 2),
                 "test_r2": round(r2_score(yB_te, pr), 3)})
resB = pd.DataFrame(resB)
print("\n[B] 점검 완료 후 '남은 일수' 회귀 (MAE 단위: 일)")
print(resB.to_string(index=False))
bestB = resB.dropna(subset=["cv_mae"]).sort_values("cv_mae").iloc[0]["model"]
gsB, pB = fittedB[bestB]
impB = permutation_importance(gsB, XB_te, yB_te, scoring="neg_mean_absolute_error", n_repeats=30, random_state=SEED)
impB = pd.Series(impB.importances_mean, index=B_CAT + B_NUM).sort_values(ascending=False).round(3)
print(f"\n[B] 선택 모델: {bestB} / 변수 중요도 (MAE 증가량, 일)\n", impB.to_string())

# ───────── 3. 저장 ─────────
resA.to_csv(f"{OUT_DIR}/model_A_classification.csv", index=False, encoding="utf-8-sig")
resB.to_csv(f"{OUT_DIR}/model_B_regression.csv", index=False, encoding="utf-8-sig")
pd.concat({"A_분류": impA, "B_회귀": impB}).to_csv(f"{OUT_DIR}/model_importance.csv", encoding="utf-8-sig")
rep.to_pickle(f"{OUT_DIR}/case_model.pkl")                    # q4에서 재사용

LABEL = {"RepairType": "수리 방식", "contact": "요청 채널", "start_hour": "접수 시각",
         "start_weekday": "접수 요일", "open_cases": "접수 시 진행 중 건수", "survey_queue": "접수 시 점검 대기 건수",
         "RepairCode": "수리 코드", "RepairInternally": "내부 수리 여부", "survey_by": "점검 담당자",
         "EstimatedRepairTime": "예상 수리시간", "survey_wait_h": "점검 대기시간", "elapsed_d": "점검까지 경과일"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
r = resA.dropna(subset=["cv_auc"])
ax[0].bar(r["model"], r["test_auc"], color=["#9aa5b1" if m != bestA else "#2f6fde" for m in r["model"]])
ax[0].axhline(0.5, ls="--", c="gray"); ax[0].text(-0.4, 0.51, "무작위 0.5", color="gray")
rule_auc = resA.loc[resA["model"] == "기준선(수리방식 규칙)", "test_auc"].iloc[0]
ax[0].axhline(rule_auc, ls="--", c="#d9534f"); ax[0].text(-0.4, rule_auc + 0.01, f"수리방식 규칙 {rule_auc:.3f}", color="#d9534f")
ax[0].set_ylim(0.4, 1); ax[0].set_title("A. 접수 직후 7일 초과 예측 (테스트 ROC-AUC)")
for i, v in enumerate(r["test_auc"]): ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
ax[1].bar(resB["model"], resB["test_mae"], color=["#c9cdd2"] + ["#9aa5b1" if m != bestB else "#2f6fde" for m in resB["model"][1:]])
ax[1].set_title("B. 점검 후 남은 일수 예측 (테스트 MAE, 일)"); ax[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(resB["test_mae"]): ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center")
ib = impB.head(6)[::-1]
ax[2].barh([LABEL.get(i, i) for i in ib.index], ib.values, color="#2f6fde")
ax[2].set_title(f"B. 변수 중요도 ({bestB}, MAE 증가량)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/fig_model.png", dpi=150)
print(f"\n저장 완료: {OUT_DIR}/model_A_classification.csv, model_B_regression.csv, model_importance.csv, fig_model.png")
