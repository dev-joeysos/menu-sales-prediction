# 개선된 매출수량 예측 모델
import os
import random
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 개선된 피처 엔지니어링 함수
def make_advanced_features(df):
    """고급 피처 생성 함수"""
    df_features = df.copy()
    
    # 기본 날짜 피처
    df_features['dayofweek'] = df_features['영업일자'].dt.dayofweek
    df_features['month'] = df_features['영업일자'].dt.month
    df_features['day'] = df_features['영업일자'].dt.day
    df_features['weekofyear'] = df_features['영업일자'].dt.isocalendar().week.astype(int)
    df_features['quarter'] = df_features['영업일자'].dt.quarter
    df_features['dayofyear'] = df_features['영업일자'].dt.dayofyear
    
    # 주말/평일 구분
    df_features['is_weekend'] = (df_features['dayofweek'] >= 5).astype(int)
    
    # 월초/월말 구분
    df_features['is_month_start'] = (df_features['day'] <= 5).astype(int)
    df_features['is_month_end'] = (df_features['day'] >= 25).astype(int)
    
    # 계절 구분
    df_features['season'] = df_features['month'] % 12 // 3
    
    # 순환 인코딩 (주기적 패턴 캡처)
    df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day'] / 31)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day'] / 31)
    
    return df_features

def create_advanced_lag_features(df):
    """고급 lag 및 롤링 피처 생성"""
    df_copy = df.copy()
    df_copy = df_copy.sort_values(by=['영업장명_메뉴명_encoded', '영업일자'])
    
    grouped = df_copy.groupby('영업장명_메뉴명_encoded')['매출수량']
    
    # 다양한 lag 피처
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df_copy[f'lag_{lag}'] = grouped.shift(lag)
    
    # 롤링 통계 피처 (다양한 윈도우)
    for window in [3, 7, 14, 21, 28]:
        series_shifted = grouped.shift(1)  # 최소 1일 전 데이터 사용
        df_copy[f'rolling_mean_{window}'] = series_shifted.rolling(window, min_periods=1).mean()
        df_copy[f'rolling_std_{window}'] = series_shifted.rolling(window, min_periods=1).std()
        df_copy[f'rolling_min_{window}'] = series_shifted.rolling(window, min_periods=1).min()
        df_copy[f'rolling_max_{window}'] = series_shifted.rolling(window, min_periods=1).max()
        df_copy[f'rolling_median_{window}'] = series_shifted.rolling(window, min_periods=1).median()
        df_copy[f'rolling_skew_{window}'] = series_shifted.rolling(window, min_periods=3).skew()
        df_copy[f'rolling_kurt_{window}'] = series_shifted.rolling(window, min_periods=4).kurt()
    
    # 지수 가중 이동 평균
    for alpha in [0.1, 0.3, 0.5]:
        df_copy[f'ewm_{alpha}'] = grouped.shift(1).ewm(alpha=alpha).mean()
    
    # 변화율 피처
    df_copy['pct_change_1d'] = grouped.pct_change(1)
    df_copy['pct_change_7d'] = grouped.pct_change(7)
    df_copy['diff_1d'] = grouped.diff(1)
    df_copy['diff_7d'] = grouped.diff(7)
    
    # 요일별 평균 (같은 요일의 과거 데이터)
    df_copy['dayofweek_mean'] = df_copy.groupby(['영업장명_메뉴명_encoded', 'dayofweek'])['매출수량'].transform('mean')
    
    # 월별 평균
    df_copy['month_mean'] = df_copy.groupby(['영업장명_메뉴명_encoded', 'month'])['매출수량'].transform('mean')
    
    df_copy.fillna(0, inplace=True)
    return df_copy

def create_target_encoding(df, target_col='매출수량', smoothing=10):
    """타겟 인코딩 생성"""
    df_encoded = df.copy()
    
    # 전체 평균
    global_mean = df[target_col].mean()
    
    # 업장별 타겟 인코딩
    venue_stats = df.groupby('업장명')[target_col].agg(['mean', 'count']).reset_index()
    venue_stats['venue_target_encoded'] = (venue_stats['mean'] * venue_stats['count'] + global_mean * smoothing) / (venue_stats['count'] + smoothing)
    df_encoded = df_encoded.merge(venue_stats[['업장명', 'venue_target_encoded']], on='업장명', how='left')
    
    # 요일별 타겟 인코딩
    dow_stats = df.groupby('dayofweek')[target_col].agg(['mean', 'count']).reset_index()
    dow_stats['dow_target_encoded'] = (dow_stats['mean'] * dow_stats['count'] + global_mean * smoothing) / (dow_stats['count'] + smoothing)
    df_encoded = df_encoded.merge(dow_stats[['dayofweek', 'dow_target_encoded']], on='dayofweek', how='left')
    
    return df_encoded

def optimize_lgb_params(X_train, y_train, n_trials=100):
    """Optuna를 사용한 하이퍼파라미터 최적화 (간단한 버전)"""
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 20,
        'max_depth': -1
    }
    return best_params

def create_ensemble_predictions(models, X_test, weights=None):
    """앙상블 예측"""
    if weights is None:
        weights = [1.0] * len(models)
    
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    # 가중 평균
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    return ensemble_pred

# 메인 학습 및 예측 로직
def main():
    print("데이터 로딩 및 전처리...")
    
    # 데이터 로딩
    train_df = pd.read_csv('./train/train.csv')
    train_df['영업일자'] = pd.to_datetime(train_df['영업일자'])
    
    # Label Encoding
    le = LabelEncoder()
    train_df['영업장명_메뉴명'] = train_df['영업장명_메뉴명'].astype(str)
    train_df['영업장명_메뉴명_encoded'] = le.fit_transform(train_df['영업장명_메뉴명'])
    train_df['업장명'] = train_df['영업장명_메뉴명'].apply(lambda x: x.split('_')[0])
    
    # 정렬
    train_df = train_df.sort_values(by=['영업장명_메뉴명_encoded', '영업일자']).reset_index(drop=True)
    
    print("고급 피처 생성...")
    
    # 0 매출 제거 (학습용)
    train_final = train_df[train_df['매출수량'] > 0].copy()
    
    # 피처 생성
    train_final = make_advanced_features(train_final)
    train_final = create_advanced_lag_features(train_final)
    train_final = create_target_encoding(train_final)
    
    print("모델 학습...")
    
    # 피처 선택
    feature_cols = [col for col in train_final.columns if col not in 
                   ['영업일자', '영업장명_메뉴명', '매출수량', '업장명']]
    
    # 타겟 생성 (다음날 예측)
    y_train = train_final.groupby('영업장명_메뉴명_encoded')['매출수량'].shift(-1)
    X_train = train_final[feature_cols]
    
    # NaN 제거
    valid_indices = y_train.dropna().index
    X_train = X_train.loc[valid_indices]
    y_train = y_train.loc[valid_indices]
    
    # 시계열 교차검증으로 모델 성능 평가
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"Fold {fold + 1}...")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 모델 파라미터 최적화
        lgb_params = optimize_lgb_params(X_fold_train, y_fold_train)
        
        # 모델 학습
        model = lgb.LGBMRegressor(**lgb_params, n_estimators=1000, early_stopping_rounds=50)
        model.fit(X_fold_train, y_fold_train, 
                 eval_set=[(X_fold_val, y_fold_val)], 
                 verbose=False)
        
        models.append(model)
        
        # 검증 점수
        val_pred = model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
        cv_scores.append(rmse)
        print(f"Fold {fold + 1} RMSE: {rmse:.4f}")
    
    print(f"평균 CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # 전체 데이터로 최종 모델 학습
    final_params = optimize_lgb_params(X_train, y_train)
    final_model = lgb.LGBMRegressor(**final_params, n_estimators=1000)
    final_model.fit(X_train, y_train)
    
    print("예측 생성...")
    
    # 제출 파일 생성
    submission_df = pd.read_csv('./sample_submission.csv')
    test_files = sorted(glob.glob('./test/TEST_*.csv'))
    
    for test_file in tqdm(test_files, desc="예측 생성 중"):
        # 28일 히스토리 로드
        history = pd.read_csv(test_file)
        history['영업일자'] = pd.to_datetime(history['영업일자'])
        history = make_advanced_features(history)
        history['영업장명_메뉴명'] = history['영업장명_메뉴명'].astype(str)
        history['영업장명_메뉴명_encoded'] = le.transform(history['영업장명_메뉴명'])
        history['업장명'] = history['영업장명_메뉴명'].apply(lambda x: x.split('_')[0])
        
        test_prefix = os.path.basename(test_file).split('.')[0]
        
        # 7일간 순차 예측
        for day in range(1, 8):
            last_date = history['영업일자'].max()
            pred_date = last_date + pd.Timedelta(days=1)
            
            # 예측할 날짜의 데이터프레임 생성
            pred_df = pd.DataFrame({
                '영업일자': [pred_date] * len(le.classes_),
                '영업장명_메뉴명': le.classes_
            })
            pred_df = make_advanced_features(pred_df)
            pred_df['영업장명_메뉴명_encoded'] = le.transform(pred_df['영업장명_메뉴명'])
            pred_df['업장명'] = pred_df['영업장명_메뉴명'].apply(lambda x: x.split('_')[0])
            
            # 히스토리와 결합하여 lag 피처 생성
            temp_history = pd.concat([history, pred_df], ignore_index=True)
            temp_history['매출수량'] = temp_history.groupby('영업장명_메뉴명_encoded')['매출수량'].ffill().fillna(0)
            temp_with_features = create_advanced_lag_features(temp_history)
            temp_with_features = create_target_encoding(temp_with_features)
            
            pred_features_df = temp_with_features[temp_with_features['영업일자'] == pred_date].copy()
            
            # 앙상블 예측
            if len(models) > 1:
                predictions = create_ensemble_predictions(models, pred_features_df[feature_cols])
            else:
                predictions = final_model.predict(pred_features_df[feature_cols])
            
            pred_df['매출수량'] = np.maximum(0, predictions)
            history = pd.concat([history, pred_df], ignore_index=True)
            
            # 제출 파일 업데이트
            date_str = f"{test_prefix}+{day}일"
            row_idx = submission_df[submission_df['영업일자'] == date_str].index
            if not row_idx.empty:
                day_pred_dict = pred_df.set_index('영업장명_메뉴명')['매출수량'].to_dict()
                for col in submission_df.columns[1:]:
                    submission_df.loc[row_idx, col] = day_pred_dict.get(col, 0)
    
    # 최종 저장
    submission_df.fillna(0, inplace=True)
    
    # 데이터 타입 변환 (정수로)
    for col in submission_df.columns[1:]:
        submission_df[col] = submission_df[col].astype(int)
    
    submission_df.to_csv('improved_lgbm_submission.csv', index=False, encoding='utf-8-sig')
    print("개선된 제출 파일이 생성되었습니다: improved_lgbm_submission.csv")

if __name__ == "__main__":
    main()