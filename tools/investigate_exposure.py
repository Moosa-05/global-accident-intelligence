import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

df = pd.read_parquet('data/processed/road_segments_exposure_accidents.parquet')
df['log_mvkt'] = np.log(df['mvkt'])

# Train/Test 80/20 split
np.random.seed(42)
shuffled = np.random.permutation(len(df))
train_idx, test_idx = shuffled[:int(0.8*len(df))], shuffled[int(0.8*len(df)):]
train, test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()

# 1. Simple baseline
rate_baseline = train['accident_count'].sum() / train['mvkt'].sum()
pred_base = test['mvkt'] * rate_baseline
mae_base = mean_absolute_error(test['accident_count'], pred_base)
rmse_base = np.sqrt(mean_squared_error(test['accident_count'], pred_base))
print(f"Simple Baseline Rate: {rate_baseline:.4f}")
print(f"  Test MAE: {mae_base:.3f}, RMSE: {rmse_base:.3f}")

# 2. Stratified rate baseline by road_class
class_rates = train.groupby('road_class')['accident_count'].sum() / train.groupby('road_class')['mvkt'].sum()
print("\nClass Rates (Train):")
for rc, r in class_rates.items():
    print(f"  {rc}: {r:.4f} crashes/MVKT")
pred_strat = test['road_class'].map(class_rates) * test['mvkt']
mae_strat = mean_absolute_error(test['accident_count'], pred_strat)
rmse_strat = np.sqrt(mean_squared_error(test['accident_count'], pred_strat))
print(f"Stratified Class Baseline Test MAE: {mae_strat:.3f}, RMSE: {rmse_strat:.3f}")

# 3. Check what was wrong with the old NB GLM
old_formula = 'accident_count ~ C(road_class) + speed_limit_kmh + latitude + longitude'
nb_old = glm(formula=old_formula, data=train, offset=train['log_mvkt'], family=sm.families.NegativeBinomial(alpha=1.0)).fit()
pred_nb_old = nb_old.predict(test, offset=test['log_mvkt'])
print(f"\nOld NB GLM Test MAE: {mean_absolute_error(test['accident_count'], pred_nb_old):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_nb_old)):.3f}")
print("Old NB params:")
print(nb_old.params)

# 5. Investigate removing coordinates, feature scaling, and alpha estimation
print("\n--- Testing Model Variations ---")

# A. Poisson GLM without coordinates (physics & network features only: road_class, speed_limit_kmh)
formula_nocoords = 'accident_count ~ C(road_class) + speed_limit_kmh'
poi_nocoords = glm(formula=formula_nocoords, data=train, offset=train['log_mvkt'], family=sm.families.Poisson()).fit()
pred_poi_nc = poi_nocoords.predict(test, offset=test['log_mvkt'])
print(f"Poisson (No Coords) MAE: {mean_absolute_error(test['accident_count'], pred_poi_nc):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_poi_nc)):.3f}")
print("Params:\n", poi_nocoords.params)

# B. Negative Binomial with estimated alpha or proper statsmodels DiscreteModel NegativeBinomial
import statsmodels.discrete.discrete_model as dm

# Use dm.NegativeBinomial (which estimates alpha via MLE!)
X_train = sm.add_constant(pd.get_dummies(train[['road_class', 'speed_limit_kmh']], drop_first=True, dtype=float))
X_test = sm.add_constant(pd.get_dummies(test[['road_class', 'speed_limit_kmh']], drop_first=True, dtype=float))
# Align columns
X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0.0)

nb_mle = dm.NegativeBinomial(train['accident_count'], X_train, offset=train['log_mvkt'], loglike_method='nb2').fit(disp=False)
pred_nb_mle = nb_mle.predict(X_test, offset=test['log_mvkt'])
print(f"\nNegative Binomial MLE (No Coords) MAE: {mean_absolute_error(test['accident_count'], pred_nb_mle):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_nb_mle)):.3f}")
print("Estimated alpha:", nb_mle.params.get('alpha', 'N/A'))
print("NB MLE params:\n", nb_mle.params)

# C. LightGBM Regressor with Poisson Objective
# Features: road_class (categorical), speed_limit_kmh, link_length_km, all_motor_vehicles, latitude, longitude
lgb_train = train.copy()
lgb_test = test.copy()
lgb_train['road_class'] = lgb_train['road_class'].astype('category')
lgb_test['road_class'] = lgb_test['road_class'].astype('category')

features = ['road_class', 'speed_limit_kmh', 'link_length_km', 'all_motor_vehicles', 'latitude', 'longitude']
# LightGBM with Poisson objective supports init_score = log(exposure)
train_data = lgb.Dataset(lgb_train[features], label=lgb_train['accident_count'], init_score=lgb_train['log_mvkt'])
valid_data = lgb.Dataset(lgb_test[features], label=lgb_test['accident_count'], init_score=lgb_test['log_mvkt'], reference=train_data)

params = {
    'objective': 'poisson',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 15,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42
}
lgb_model = lgb.train(params, train_data, num_boost_round=100)
pred_lgb = lgb_model.predict(lgb_test[features]) * lgb_test['mvkt']  # Note: in LightGBM poisson with init_score, predict returns raw exp(pred + init_score) = rate * mvkt
# wait! With init_score, predict() already includes exp(init_score)! Let's verify:
pred_lgb_raw = lgb_model.predict(lgb_test[features])
print(f"\nLightGBM (Poisson objective) Test MAE: {mean_absolute_error(test['accident_count'], pred_lgb_raw):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_lgb_raw)):.3f}")

# D. Zero-Inflated Poisson / Zero-Inflated Negative Binomial
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP
try:
    zip_model = ZeroInflatedPoisson(train['accident_count'], X_train, offset=train['log_mvkt']).fit(disp=False)
    pred_zip = zip_model.predict(X_test, offset=test['log_mvkt'])
    print(f"\nZero-Inflated Poisson MAE: {mean_absolute_error(test['accident_count'], pred_zip):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_zip)):.3f}")
except Exception as e:
    print(f"\nZero-Inflated Poisson failed: {e}")

try:
    zinb_model = ZeroInflatedNegativeBinomialP(train['accident_count'], X_train, offset=train['log_mvkt']).fit(disp=False)
    pred_zinb = zinb_model.predict(X_test, offset=test['log_mvkt'])
    print(f"\nZero-Inflated Negative Binomial MAE: {mean_absolute_error(test['accident_count'], pred_zinb):.3f}, RMSE: {np.sqrt(mean_squared_error(test['accident_count'], pred_zinb)):.3f}")
except Exception as e:
    print(f"\nZero-Inflated NB failed: {e}")

