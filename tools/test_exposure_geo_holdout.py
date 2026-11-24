import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

df = pd.read_parquet('data/processed/road_segments_exposure_accidents.parquet')
df['log_mvkt'] = np.log(df['mvkt'])

# Define a strict geographic holdout:
# UK Northern & Celtic regions: Latitude >= 54.5 (Scotland + Northern England) or Longitude <= -3.5 (Wales / Cornwall)
is_spatial_holdout = (df['country_code'] == 'GB') & ((df['latitude'] >= 54.5) | (df['longitude'] <= -3.5))

train_df = df[~is_spatial_holdout].copy()
test_geo = df[is_spatial_holdout].copy()

print(f"Total Segments: {len(df)}")
print(f"Training Segments (England Midlands/South & France): {len(train_df)}")
print(f"Geographic Holdout Segments (Scotland, Wales, North): {len(test_geo)}")

# 1. Simple Baseline
rate_base = train_df['accident_count'].sum() / train_df['mvkt'].sum()
pred_base = test_geo['mvkt'] * rate_base
mae_base = mean_absolute_error(test_geo['accident_count'], pred_base)
rmse_base = np.sqrt(mean_squared_error(test_geo['accident_count'], pred_base))
print(f"\n[Baseline] Mean Rate: {rate_base:.4f} crashes/MVKT")
print(f"  Geo Holdout MAE: {mae_base:.3f}, RMSE: {rmse_base:.3f}")

# 2. Stratified Road Class Baseline
strat_rates = train_df.groupby('road_class')['accident_count'].sum() / train_df.groupby('road_class')['mvkt'].sum()
pred_strat = test_geo['road_class'].map(strat_rates) * test_geo['mvkt']
mae_strat = mean_absolute_error(test_geo['accident_count'], pred_strat)
rmse_strat = np.sqrt(mean_squared_error(test_geo['accident_count'], pred_strat))
print(f"\n[Stratified Baseline] Geo Holdout MAE: {mae_strat:.3f}, RMSE: {rmse_strat:.3f}")

# 3. Poisson GLM (No spatial coords)
poi_formula = 'accident_count ~ C(road_class) + speed_limit_kmh'
poi_model = glm(formula=poi_formula, data=train_df, offset=train_df['log_mvkt'], family=sm.families.Poisson()).fit()
pred_poi = poi_model.predict(test_geo, offset=test_geo['log_mvkt'])
mae_poi = mean_absolute_error(test_geo['accident_count'], pred_poi)
rmse_poi = np.sqrt(mean_squared_error(test_geo['accident_count'], pred_poi))
print(f"\n[Poisson GLM] Geo Holdout MAE: {mae_poi:.3f}, RMSE: {rmse_poi:.3f}")

# 4. LightGBM Poisson (Without raw spatial coords to prevent spatial memorization)
# Only physics & network features: road_class, speed_limit_kmh, link_length_km, all_motor_vehicles
features = ['road_class', 'speed_limit_kmh', 'link_length_km', 'all_motor_vehicles']
train_lgb = train_df.copy()
test_lgb = test_geo.copy()
train_lgb['road_class'] = train_lgb['road_class'].astype('category')
test_lgb['road_class'] = test_lgb['road_class'].astype('category')

train_data = lgb.Dataset(train_lgb[features], label=train_lgb['accident_count'], init_score=train_lgb['log_mvkt'])
valid_data = lgb.Dataset(test_lgb[features], label=test_lgb['accident_count'], init_score=test_lgb['log_mvkt'], reference=train_data)

params = {
    'objective': 'poisson',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 12,
    'min_data_in_leaf': 30,
    'verbose': -1,
    'seed': 42
}
lgb_model = lgb.train(params, train_data, num_boost_round=100)
pred_lgb = lgb_model.predict(test_lgb[features])
mae_lgb = mean_absolute_error(test_geo['accident_count'], pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(test_geo['accident_count'], pred_lgb))
print(f"\n[LightGBM Poisson (No Coords)] Geo Holdout MAE: {mae_lgb:.3f}, RMSE: {rmse_lgb:.3f}")

# 5. Calibration: Observed vs Predicted on Geo Holdout
print("\n--- Geographic Holdout Calibration by Road Class ---")
for rc in ['MOTORWAY', 'PRIMARY_ARTERIAL']:
    mask = test_geo['road_class'] == rc
    obs_count = test_geo.loc[mask, 'accident_count'].sum()
    obs_mvkt = test_geo.loc[mask, 'mvkt'].sum()
    obs_rate = obs_count / obs_mvkt
    pred_base_rate = pred_base[mask].sum() / obs_mvkt
    pred_poi_rate = pred_poi[mask].sum() / obs_mvkt
    pred_lgb_rate = pred_lgb[mask].sum() / obs_mvkt
    print(f"Class: {rc} (Segments: {mask.sum()})")
    print(f"  Observed Rate: {obs_rate:.4f} crashes/MVKT (Total Crashes: {obs_count})")
    print(f"  Baseline Pred Rate: {pred_base_rate:.4f} (Total: {pred_base[mask].sum():.0f})")
    print(f"  Poisson Pred Rate: {pred_poi_rate:.4f} (Total: {pred_poi[mask].sum():.0f})")
    print(f"  LightGBM Pred Rate: {pred_lgb_rate:.4f} (Total: {pred_lgb[mask].sum():.0f})")
