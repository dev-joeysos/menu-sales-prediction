# 식음업장 메뉴 수요 예측 — LG Aimers 7기 온라인 해커톤

> LG AI 연구원 주최 · 데이콘 주관 · 2025.08.01 ~ 08.25
> [대회 페이지](https://dacon.io/competitions/official/236559/overview/description)

## 결과

| 항목 | 값 |
|---|---|
| 최종 순위 | **281위** (참가자 1,779명) |
| 점수 | **0.59373** (SMAPE 기반, 낮을수록 좋음) |
| 제출 횟수 | 16회 |
| 팀 | 후이후이 (개인 참가) |

## 과제

리조트 식음업장의 과거 **메뉴별 일 판매량**으로 **향후 1주일(7일) 판매량**을 메뉴마다 예측합니다.
테스트 파일(`TEST_00` ~ `TEST_09`)마다 직전 기간 데이터가 주어지고, 이어지는 7일을 예측해 제출합니다.

## 접근 과정

| 날짜 | 단계 | 내용 | 파일 |
|---|---|---|---|
| ~08.13 | 베이스라인 | 대회 제공 LSTM 베이스라인 실행 | `Baseline.ipynb` |
| 08.13 | LightGBM 전환 | 메뉴 단위 LightGBM 회귀. 날짜 피처(요일·월·주차·주말), `영업장명_메뉴명` 인코딩. 1일 예측 모델을 7일간 **자기회귀(autoregressive)** 로 반복 적용 | `매출수량예측.ipynb` |
| 08.13 | 모델 개선 실험 | 고급 날짜 피처, 지연(lag)·이동 통계 피처, 타깃 인코딩, **Optuna** 하이퍼파라미터 탐색, 앙상블 | `improved_sales_prediction.py` |
| 08.18 | 학습 데이터 조정 | 판매량 0인 날도 학습에 포함, 특정 영업장 가중치 조정 | 커밋 `66211c3`, `7ec70b3` |
| 08.18 | 캘린더·이벤트 피처 | 공휴일·징검다리 휴일·연휴, 학기/방학 여부, 공휴일까지 거리, 요일·연중일자의 sin/cos 주기 인코딩 | `매출수량예측_calendar_fixed.ipynb` |

## 파일 구성

```
Baseline.ipynb                      대회 베이스라인 (LSTM)
매출수량예측.ipynb                    LightGBM + 자기회귀 예측
매출수량예측_calendar_fixed.ipynb      캘린더·이벤트 피처를 추가한 최종 노트북
improved_sales_prediction.py        피처 확장 + Optuna + 앙상블 실험 스크립트
```

대회 규정상 데이터(`train/`, `test/`)와 제출 파일(`*.csv`)은 저장소에 포함하지 않습니다(`.gitignore`).

## 실행

1. 데이콘에서 대회 데이터를 받아 `train/train.csv`, `test/TEST_00~09.csv`, `sample_submission.csv`를 이 폴더에 둡니다.
2. `매출수량예측_calendar_fixed.ipynb`를 위에서부터 실행하면 `lgbm_submission.csv`가 만들어집니다.

## 회고

<!-- TODO: 직접 작성 — 잘 된 점, 아쉬운 점, 다음에 시도할 것 -->
