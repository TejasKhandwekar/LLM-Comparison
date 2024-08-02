# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np


from neuralforecast import NeuralForecast
from neuralforecast.models import (
    NBEATS,
    NHITS,
    TFT)

from statsforecast.core import StatsForecast
from neuralforecast.auto import AutoMLP, AutoDeepAR, AutoNBEATS, AutoNHITS, AutoTFT , AutoDeepNPTS
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    ARIMA,
    SimpleExponentialSmoothing,
    SimpleExponentialSmoothingOptimized,
    SeasonalExponentialSmoothing,
    SeasonalExponentialSmoothingOptimized,
    RandomWalkWithDrift,
    ETS,
    HistoricAverage,
    WindowAverage,
    AutoARIMA,
    AutoETS,
    AutoCES,
    AutoTheta
)
from statsforecast.models import SimpleExponentialSmoothing
# Naive
# Naive season
# Seasonal Arima (0,1 1)( 0,1,1)

from mlforecast import MLForecast
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
from mlforecast.target_transforms import Differences
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from numba import njit
from window_ops.rolling import rolling_mean

import time
import plotly.graph_objects as go


# %%
def evaluate_forecast(y_true, y_pred):
    return np.sqrt(np.mean((y_true.values - y_pred.values) ** 2))

def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)) / 2)

def rmsse(y_true, y_pred, train):
    naive_forecast = np.roll(train, 1)
    naive_forecast[0] = train[0]
    scale = np.mean((train - naive_forecast) ** 2)
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)

def calculate_errors(test, forecasts, train):

    df = pd.DataFrame()
    for col in forecasts.columns:
        error_dict = {}
        
        if col in ['ds', 'unique_id']:
            continue
        else:
            y_true = test['y'].values
            y_pred = forecasts[col].values
            error_dict.update({
                f'RMSE': evaluate_forecast(test['y'], forecasts[col]),
                f'SMAPE': smape(y_true, y_pred),
                f'RMSSE': rmsse(y_true, y_pred, train['y'].values)
            })
        df = pd.concat([pd.DataFrame(error_dict, index=[col]),df])
    return df


# %%
from statsforecast.utils import AirPassengersDF

# Load the AirPassengers dataset
df = AirPassengersDF

forecast_horizon = [24, 3,6,12]

# Train-test split
test_size_total = 24

train_size_total = len(df) - test_size_total
train_total, test_total = df[:train_size_total], df[train_size_total:]


df.to_csv('Air_passengers.csv')


# %%
train


# %%
models = [
    Naive(),
    SeasonalNaive(12),
    ARIMA(order=[12,1,0]),
    ARIMA(order=[0,1,1], seasonal_order=[0,1,1], season_length=12 ,alias='SARIMA'),
    SimpleExponentialSmoothing(alpha=0.28),
    ETS(model='AAA',season_length=12, alias='ETS AAA'),
    ETS(model='MAM',season_length=12, alias='ETS MAM'),
    ETS(model='MMM',season_length=12, alias='ETS MMM'),
    ETS(model='MAM',season_length=12, alias='ETS MAdM',damped=True),
    HistoricAverage(),
    WindowAverage(window_size=6),
    AutoARIMA(max_p=12),
    AutoETS(season_length=12),
    AutoETS(season_length=12,damped=True,alias='Damped AutoETS'),
    AutoCES(season_length=12,alias='AutoCES'),
    AutoTheta(season_length=12),
    SimpleExponentialSmoothingOptimized(),
    SeasonalExponentialSmoothing(season_length=12,alpha=0.28),
    SeasonalExponentialSmoothingOptimized(season_length=12),
    RandomWalkWithDrift(),
]

error_dfs = []

for model in models:
    for horizon in forecast_horizon:
        df_forecast_model = pd.DataFrame()
        total_train_time = 0
        combined_forecasts = pd.DataFrame()

        for start in range(0, test_size_total, horizon):
            end = start + horizon
            train_size = train_size_total + start
            train = df[:train_size]
            test = df[train_size:train_size + horizon]
            sf = StatsForecast(
                models=[model],
                freq='ME',
                n_jobs=-1,
            )

            start_time = time.time()
            forecasts_df = sf.forecast(df=train, h=horizon)
            train_time = time.time() - start_time

            forecasts_df['origin'] = train_size  # Track the forecast origin point
            combined_forecasts = pd.concat([combined_forecasts, forecasts_df])
            total_train_time += train_time
        
            
        # Calculate errors for the combined forecast
        combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
        
        combined_error_statsforecast['Total_Train_Time'] = total_train_time
        combined_error_statsforecast['Horizon'] = horizon
        combined_error_statsforecast['Model'] = str(model)
        error_dfs.append(combined_error_statsforecast)

# Combine all errors into a single DataFrame
Error_statstical = pd.concat(error_dfs).reset_index(drop=True)


# %%
@njit
def rolling_mean_12(x):
    return rolling_mean(x, window_size=12)


def month_index(times):
    return times.month


# %%
lgb_params = {'verbosity': -1,'num_leaves': 512,}

catboost_params ={'subsample': 0.6 , 'iterations': 50, 'depth': 5, 'verbose':0}

xgboost_params ={'verbosity':0, 'max_depth':5 , 'subsample': 0.6}

randomforest_params = {'verbose': 0, 'max_depth': 5}
models={
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
        'CatBoost': CatBoostRegressor(**catboost_params),
        'XgBoost': XGBRegressor(**xgboost_params),
        'RandomForest': RandomForestRegressor(**randomforest_params)
    }


# %%
lgb_params = {'verbosity': -1,'num_leaves': 512,}

catboost_params ={'subsample': 0.6 , 'iterations': 50, 'depth': 5, 'verbose':0}

xgboost_params ={'verbosity':0, 'max_depth':5 , 'subsample': 0.6}

randomforest_params = {'verbose': 0, 'max_depth': 5}
models={
        'LightGBM': lgb.LGBMRegressor(**lgb_params),
        'CatBoost': CatBoostRegressor(**catboost_params),
        'XgBoost': XGBRegressor(**xgboost_params),
        'RandomForest': RandomForestRegressor(**randomforest_params)
    }

for alias,model in models.items():
    for horizon in forecast_horizon:
        df_forecast_model = pd.DataFrame()
        total_train_time = 0
        combined_forecasts = pd.DataFrame()

        for start in range(0, test_size_total, horizon):
            end = start + horizon
            train_size = train_size_total + start
            train = df[:train_size]
            test = df[train_size:train_size + horizon]
            
            fcst = MLForecast(
                models = {alias:model,},
                freq="ME",
                target_transforms=[Differences([12])],    
                lags= [1,2,3,4,11,12],
                lag_transforms={
                    1: [ExpandingMean()],
                    12: [RollingMean(window_size=12), rolling_mean_12],
                },
                date_features=[month_index],
            )

            start_time = time.time()
            prep = fcst.preprocess(train)
            fcst.fit(train)
            forecasts_df = fcst.predict(h=horizon)
            train_time = time.time() - start_time

            forecasts_df['origin'] = train_size  # Track the forecast origin point
            combined_forecasts = pd.concat([combined_forecasts, forecasts_df])
            total_train_time += train_time
        
            
        # Calculate errors for the combined forecast
        combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
        
        combined_error_statsforecast['Total_Train_Time'] = total_train_time
        combined_error_statsforecast['Horizon'] = horizon
        combined_error_statsforecast['Model'] = alias
        error_dfs.append(combined_error_statsforecast)

# Combine all errors into a single DataFrame
Error_Tree = pd.concat(error_dfs).reset_index(drop=True)


# %%
Error_Tree


# %%
from ray import tune
from ray import tune
neural_models = [
    NBEATS(input_size=2 * test_size_total, h=test_size_total,),
    NHITS(input_size=2 * test_size_total, h=test_size_total,),
    AutoMLP(config=dict(input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, cpus=3,verbose=False),
    AutoDeepAR(config=dict( input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, cpus=3),
    AutoNBEATS(config=dict( input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, cpus=3),
    AutoNHITS(config=dict(input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, cpus=3),
    AutoTFT(config=dict( input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, cpus=3)
]



for horizon in forecast_horizon:
    for model in neural_models:
        total_train_time = 0
        combined_forecasts = pd.DataFrame()
        for start in range(0, test_size_total, horizon):
            end = start + horizon
            train_size = train_size_total + start
            train = df[:train_size]
            test = df[train_size:train_size + horizon]

            nf = NeuralForecast(models=[model], freq='ME')

            start_time = time.time()
            nf.fit(df=train)
            forecasts_df_neural = nf.predict().reset_index()
            train_time = time.time() - start_time
            forecasts_df_neural = forecasts_df_neural[:horizon]
            forecasts_df_neural['origin'] = train_size  # Track the forecast origin point
            combined_forecasts = pd.concat([combined_forecasts, forecasts_df_neural])
            total_train_time += train_time

        # Calculate errors for the combined forecast
        combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
        combined_error_statsforecast['Total_Train_Time'] = total_train_time
        combined_error_statsforecast['Horizon'] = horizon
        combined_error_statsforecast['Model'] = type(model).__name__ + '_old'
        error_dfs.append(combined_error_statsforecast)

# Combine all errors into a single DataFrame
all_errors_old = pd.concat(error_dfs).reset_index(drop=True)
print(all_errors_old)


# %%
# error_dfs=[]
# forecast_horizon = [24, 3]
# for model in neural_models:
#     for horizon in forecast_horizon:
#         total_train_time = 0
#         combined_forecasts = pd.DataFrame()
#         for start in range(0, test_size_total, horizon):
#             end = start + horizon
#             train_size = train_size_total + start
#             train = df[:train_size]
#             test = df[train_size:train_size + horizon]

#             nf = NeuralForecast(models=[model], freq='ME')

#             start_time = time.time()
#             nf.fit(df=train,verbose=False)
#             forecasts_df_neural = nf.predict().reset_index()
#             train_time = time.time() - start_time
#             forecasts_df_neural = forecasts_df_neural[:horizon]
#             forecasts_df_neural['origin'] = train_size  # Track the forecast origin point
#             combined_forecasts = pd.concat([combined_forecasts, forecasts_df_neural])
#             total_train_time += train_time

#         # Calculate errors for the combined forecast
#         combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
#         combined_error_statsforecast['Total_Train_Time'] = total_train_time
#         combined_error_statsforecast['Horizon'] = horizon
#         combined_error_statsforecast['Model'] = type(model).__name__
#         error_dfs.append(combined_error_statsforecast)

# # Combine all errors into a single DataFrame
# all_errors_old = pd.concat(error_dfs).reset_index(drop=True)
# print(all_errors_old)


# %%
for horizon in forecast_horizon:
    neural_models = [
      NBEATS(input_size=2 * horizon, h=horizon,),
      NHITS(input_size=2 * horizon, h=horizon,),
      AutoMLP(config=dict(input_size=tune.choice([3 * horizon]),      learning_rate=tune.choice([1e-3])), h=horizon, num_samples=1,verbose=False),
      AutoDeepAR(config=dict( input_size=tune.choice([3 * horizon]), learning_rate=tune.choice([1e-3])), h=horizon, num_samples=1,),
      AutoNBEATS(config=dict( input_size=tune.choice([3 * horizon]), learning_rate=tune.choice([1e-3])), h=horizon, num_samples=1,),
      AutoNHITS(config=dict(input_size=tune.choice([3 * horizon]), learning_rate=tune.choice([1e-3])), h=horizon, num_samples=1,),
      AutoTFT(config=dict( input_size=tune.choice([3 * horizon]), learning_rate=tune.choice([1e-3])), h=horizon, num_samples=1,)
    ]
    for model in neural_models:
        total_train_time = 0
        combined_forecasts = pd.DataFrame()
        for start in range(0, test_size_total, horizon):
            end = start + horizon
            train_size = train_size_total + start
            train = df[:train_size]
            test = df[train_size:train_size + horizon]

            nf = NeuralForecast(models=[model], freq='ME')

            start_time = time.time()
            nf.fit(df=train)
            forecasts_df_neural = nf.predict().reset_index()
            train_time = time.time() - start_time
            forecasts_df_neural = forecasts_df_neural[:horizon]
            forecasts_df_neural['origin'] = train_size  # Track the forecast origin point
            combined_forecasts = pd.concat([combined_forecasts, forecasts_df_neural])
            total_train_time += train_time

        # Calculate errors for the combined forecast
        combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
        combined_error_statsforecast['Total_Train_Time'] = total_train_time
        combined_error_statsforecast['Horizon'] = horizon
        combined_error_statsforecast['Model'] = type(model).__name__ + 'new'
        error_dfs.append(combined_error_statsforecast)

# Combine all errors into a single DataFrame
all_errors_new = pd.concat(error_dfs).reset_index(drop=True)
print(all_errors_new)


# %%
from nixtla import NixtlaClient
nixtla_client = NixtlaClient(
    api_key = 'nixtla-tok-BWWtvgUP9FLtzerA90xyzXPvRUoZvA0OYYp5cuSI7NZUyApQjlINlF8dAyYXqDyxWlTlCOg7jXHWJV4o'
)


# %%
# error_dfs = []
for horizon in forecast_horizon:
    df_forecast_model = pd.DataFrame()
    total_train_time = 0
    combined_forecasts = pd.DataFrame()

    for start in range(0, test_size_total, horizon):
        end = start + horizon
        train_size = train_size_total + start
        train = df[:train_size]
        test = df[train_size:train_size + horizon]
        sf = StatsForecast(
            models=[model],
            freq='ME',
            n_jobs=-1,
        )

        start_time = time.time()
        forecasts_df = nixtla_client.forecast(df=train, h=horizon, freq='M', time_col='ds', target_col='y')
        train_time = time.time() - start_time

        forecasts_df['origin'] = train_size  # Track the forecast origin point
        combined_forecasts = pd.concat([combined_forecasts, forecasts_df])
        total_train_time += train_time
    
        
    # Calculate errors for the combined forecast
    combined_error_statsforecast = calculate_errors(test_total, combined_forecasts.drop(columns=['origin']), train_total)
    
    combined_error_statsforecast['Total_Train_Time'] = total_train_time
    combined_error_statsforecast['Horizon'] = horizon
    combined_error_statsforecast['Model'] = "TimeGPT"
    error_dfs.append(combined_error_statsforecast)

# Combine all errors into a single DataFrame
Error_TimeGPT = pd.concat(error_dfs).reset_index(drop=True)


# %%
# Sort the DataFrame by the 'Error' column
sorted_df = Error_TimeGPT.sort_values(by='RMSSE')



sorted_df.to_csv('results.csv')


