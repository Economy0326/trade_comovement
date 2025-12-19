# 제3회 국민대학교 AI빅데이터 분석 경진대회  
## 공행성(pair) 판별 및 다음달 무역량 예측 모델 최종 결과보고서

---

## 1. 프로젝트 개요

### 1.1 문제 정의
본 프로젝트는 국민대학교 경영대학원과 한국기계산업진흥회(KOAMI)가 공동 주최한  
**「제3회 국민대학교 AI빅데이터 분석 경진대회」** 예선 과제로,  
100개 수입 품목의 월별 무역 데이터를 기반으로 다음 두 가지 문제를 해결하는 것을 목표로 한다.

1. **공행성(pair) 판별**  
   - 선행 품목(`leading_item_id`)과 후행 품목(`following_item_id`) 간  
     시간 지연(lag)을 두고 연동되는 구조적 관계가 존재하는 품목 쌍을 탐색

2. **다음달 무역량 예측**  
   - 선행 품목의 흐름을 활용하여  
     2025년 8월 후행 품목의 무역량(`value`)을 예측

---

### 1.2 평가 방식 요약
예선 평가는 다음의 복합 지표로 이루어진다.
```
Score = 0.6 × F1 + 0.4 × (1 − NMAE)
```
- **F1-score**: 공행성 pair 판별 성능
- **NMAE**: 다음달 무역량 예측 오차 지표
- FP 또는 FN에 해당하는 pair는 NMAE에서 **오차 1.0(최하점)** 처리됨

➡️ 따라서 본 문제는  
**pair 선택 정확도(F1)와 회귀 예측 안정성(NMAE)을 동시에 고려해야 하는 문제**이다.

---

## 2. 데이터 이해 및 전처리

### 2.1 데이터 구성
`train.csv`는 다음과 같은 구조를 가진다.

- `item_id`: 품목 식별자 (총 100개)
- `year`, `month`: 연도 및 월
- `seq`: 동일 연-월 내 일련번호
- `type`, `hs4`: 품목 속성
- `weight`, `quantity`, `value`: 무역량 관련 변수

본 프로젝트에서는 **월별 무역량(value)** 예측이 목표이므로,  
value 컬럼을 중심으로 데이터 전처리를 수행하였다.

---

### 2.2 월별 집계 및 Pivot 생성
1. `(item_id, year, month)` 기준으로 `value` 합계 집계
2. `item_id × 월(ym)` 형태의 pivot 테이블 생성
3. 결측 월은 0으로 대체

이로써 각 품목은 월별 무역량 시계열로 표현된다.

---

## 3. 전체 접근 전략

본 문제의 가장 큰 특징은 **공행성 pair에 대한 정답 레이블이 제공되지 않는다는 점**이다.  
이에 따라 본 프로젝트는 다음과 같은 전략을 채택하였다.

1. **시차 기반 상관 분석**을 통해 공행성 후보 pair 생성
2. 상관이 큰 pair를 이용해 **pseudo-label(약지도)** 생성
3. **분류 모델(XGBClassifier)**로 공행성 확률 추정
4. 선택된 pair에 대해 **회귀 모델(XGBRegressor)**로 다음달 무역량 예측

즉, 본 모델은  
**Pair Classification → Pair Selection → Value Regression**의 2단계 구조로 설계되었다.

---

## 4. 공행성(pair) 판별 알고리즘

### 4.1 시차(lag) 기반 상관 분석
선행 품목 A와 후행 품목 B 간 공행성은  
다음과 같은 시차 상관으로 정의하였다.
```
corr(A[t], B[t + lag]), lag = 1 ~ MAX_LAG
```
각 pair에 대해 다음 통계 피처를 계산하였다.

- `max_corr`: lag별 상관 중 절대값 최대
- `best_lag`: max_corr가 발생한 lag
- `second_corr`: 두 번째로 큰 상관
- `corr_stability = |max_corr − second_corr|`
- `corr_mean`, `corr_std`, `corr_abs_mean`

이를 통해 각 pair를 하나의 feature 벡터로 요약하였다.

---

### 4.2 Pseudo-label 생성
정답 pair가 없으므로, 다음 기준으로 약지도(pseudo-label)를 생성하였다.

- `label = 1` if `abs(max_corr) ≥ threshold`
- `label = 0` otherwise

하이퍼파라미터:
- `PAIR_LABEL_CORR_THRESHOLD = 0.38`
- 희소성 노이즈 제거를 위해 non-zero 개수가 적은 품목은 제외

클래스 불균형 완화를 위해 negative sampling을 적용하였다.

---

### 4.3 공행성 분류기 학습
pseudo-label 데이터를 이용하여 **XGBClassifier**를 학습하였다.

- 입력: 상관 기반 통계 피처 + 희소성/규모 관련 피처
- 출력: 해당 pair가 공행성일 확률(`clf_prob`)

이를 통해 단순 상관 임계값 기반 판단보다  
더 유연한 공행성 판단을 수행하도록 설계하였다.

---

### 4.4 최종 pair 선택 전략 (Tau + Backfill)
공행성 pair 선택 시 다음 전략을 사용하였다.

1. `clf_prob ≥ PROB_TAU`인 pair 우선 선택
2. 선택된 pair 수가 목표 K보다 부족할 경우,
   확률 상위 pair로 **backfill하여 항상 K개 유지**

이 전략을 통해:
- 확률 threshold로 FP를 줄이면서
- FN 폭증으로 인한 F1 급락을 방지하였다.

---

## 5. 다음달 무역량 예측 모델

### 5.1 회귀 문제 정의
선택된 pair(A → B)에 대해  
마지막 관측월을 기준으로 **B의 다음달 무역량(value)**을 예측한다.

---

### 5.2 회귀 피처 구성
회귀 피처는 다음 세 영역으로 구성된다.

1. **후행 품목(B)의 최근 흐름**
   - 최근 3개월 값, 이동평균, 변화율

2. **선행 품목(A)의 lag 반영 값**
   - lag 적용 값, 이동평균, 변화율

3. **pair 관계 피처**
   - 상관 계수, best_lag, 안정성 지표

---

### 5.3 XGBRegressor 및 Baseline Blending
회귀 모델로 **XGBRegressor**를 사용하였다.

또한 예측 안정화를 위해,
후행 품목의 최근 3개월 평균(MA3)을 baseline으로 활용하여
다음과 같이 블렌딩하였다.
```
pred_final = α × pred_model + (1 − α) × pred_baseline
```

- `α = 0.95`

이를 통해 이상치 예측으로 인한 NMAE 악화를 완화하였다.

---

## 6. 구현 코드 구조

주요 함수 구성은 다음과 같다.

- `load_pivot()` : 월별 pivot 생성
- `calc_pair_stats()` : lag 기반 상관 통계 계산
- `build_pair_feature_matrix()` : pseudo-label 생성
- `train_pair_classifier()` : 공행성 분류기 학습
- `score_all_pairs()` : 전체 pair 확률 산출
- `build_reg_dataset()` : 회귀 학습 데이터 구성
- `train_regressor()` : 무역량 예측 모델 학습
- `run_submission()` : 전체 파이프라인 실행 및 제출 파일 생성

---

## 7. 실험 및 개선 과정

- 단순 상관 기반 pair 선택 → FP 다수 발생
- pseudo-label + 분류기 도입 → 구조적 관계 학습 가능
- 확률 threshold 적용 시 FN 폭증 문제 발생
- **Tau + Backfill 전략 도입으로 F1 안정화**
- 회귀 모델에 baseline blending 적용으로 NMAE 개선

---

## 8. 성능 평가 결과

| 방법 | Public Score |
|----|----|
| sample_submission baseline | ~0.14 |
| 단순 상관 기반 | ~0.20 |
| pseudo-label + 분류기 | ~0.26 |
| **최종 모델** | **~0.30** |

---

## 9. 한계 및 향후 개선 방향

- 공통 추세로 인한 가짜 상관 제거 필요
- 선행/후행 방향성 검증 강화
- pseudo-label 생성 시 시간적 안정성 반영
- 회귀 target log 변환 및 예측값 clip 적용

---

## 10. 결론

본 프로젝트는  
**시차 기반 상관 분석과 머신러닝을 결합하여  
무역 품목 간 공행성 구조를 탐색하고,  
이를 활용한 다음달 무역량 예측 모델을 구현**하였다.

정답 pair가 없는 환경에서 pseudo-label을 활용한 접근을 통해  
baseline 대비 의미 있는 성능 향상을 달성하였다.
