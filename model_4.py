# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %%
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import pickle
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, TFT
from neuralforecast.auto import AutoMLP, AutoDeepAR, AutoNBEATS, AutoNHITS, AutoTFT, AutoDeepNPTS
from neuralforecast.losses.pytorch import DistributionLoss, HuberMQLoss
from statsforecast.core import StatsForecast
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
from tqdm import tqdm
from mlforecast import MLForecast
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

from mlforecast.target_transforms import Differences
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from numba import njit
from window_ops.rolling import rolling_mean

from ray import tune
from nixtla import NixtlaClient
import contextlib
import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

desired_width=320

pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns',10)


# %%
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
        if col in ['ds', 'unique_id']:
            continue
        y_true = test['y'].values
        y_pred = forecasts[col].values
        error_dict = {
            'RMSE': evaluate_forecast(test['y'], forecasts[col]),
            'SMAPE': smape(y_true, y_pred),
            'RMSSE': rmsse(y_true, y_pred, train['y'].values)
        }
        df = pd.concat([pd.DataFrame(error_dict, index=[col]), df])
    return df

# %%
# Load the AirPassengers dataset
from statsforecast.utils import AirPassengersDF
df = AirPassengersDF

# forecast_horizons = [3,6,12]
test_size_total = 24
train_size_total = len(df) - test_size_total
train_total, test_total = df[:train_size_total], df[train_size_total:]

df.to_csv('Air_passengers.csv')


# %%
# %%
@njit
def rolling_mean_12(x):
    return rolling_mean(x, window_size=12)

@njit
def rolling_mean_24(x):
    return rolling_mean(x, window_size=24)

def month_index(times):
    return times.month


# %%
# %%
# Model definitions
statistical_models = [
    Naive(),
    SeasonalNaive(12),
    ARIMA(order=[12, 1, 0]),
    ARIMA(order=[0, 1, 1], seasonal_order=[0, 1, 1], season_length=12, alias='SARIMA'),
    SimpleExponentialSmoothing(alpha=0.28),
    ETS(model='AAA', season_length=12, alias='ETS AAA'),
    ETS(model='MAM', season_length=12, alias='ETS MAM'),
    ETS(model='MMM', season_length=12, alias='ETS MMM'),
    ETS(model='MMM', season_length=12, alias='ETS MMdM', damped=True),
    ETS(model='MAM', season_length=12, alias='ETS MAdM', damped=True),
    HistoricAverage(),
    WindowAverage(window_size=6),
    AutoARIMA(max_p=12),
    AutoETS(season_length=12),
    AutoETS(season_length=12, damped=True, alias='Damped AutoETS'),
    AutoCES(season_length=12, alias='AutoCES'),
    AutoTheta(season_length=12),
    SimpleExponentialSmoothingOptimized(),
    SeasonalExponentialSmoothing(season_length=12, alpha=0.28),
    SeasonalExponentialSmoothingOptimized(season_length=12),
    RandomWalkWithDrift()
]

lgb_params = {'verbosity': -1, 'num_leaves': 512}
catboost_params = {'subsample': 0.6, 'iterations': 50, 'depth': 5, 'verbose': 0}
xgboost_params = {'verbosity': 0, 'max_depth': 5, 'subsample': 0.6}
randomforest_params = {'verbose': 0, 'max_depth': 5}

tree_models = [
    [{'LightGBM': lgb.LGBMRegressor(**lgb_params)}],
    [{'CatBoost': CatBoostRegressor(**catboost_params)}],
    [{'XgBoost': XGBRegressor(**xgboost_params)}],
    [{'RandomForest': RandomForestRegressor(**randomforest_params)}],
]

neural_models_template = [
    NBEATS(input_size=2 * test_size_total, h=test_size_total,max_steps=100),
    NHITS(input_size=2 * test_size_total, h=test_size_total,max_steps=100),
    AutoMLP(config=dict(max_steps=100,input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1,verbose=False),
    AutoDeepAR(config=dict(max_steps=100, input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1,),
    AutoNBEATS(config=dict(max_steps=100, input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, ),
    AutoNHITS(config=dict(max_steps=100,input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1, ),
    AutoTFT(config=dict( max_steps=100,input_size=tune.choice([3 * test_size_total]), learning_rate=tune.choice([1e-3])), h=test_size_total, num_samples=1,)
]

from nixtla import NixtlaClient
nixtla_client = NixtlaClient(
    api_key = 'nixtla-tok-1UeN6TY9k2Nn1GCoqyDkCG8daFEoHYOwIZuohMOU99Wa116AvDQF9rKjNSqVVIBuuWXkXpiwW5v3tlMb'
)


# %%
# %%
forecast_horizons = [i for i in range(1,19)]
def log_to_file(message):
    with open("model_training_log.txt", "a") as file:
        file.write(message + "\n")
def forecast_and_evaluate_overlapping(models, model_type, forecast_horizons, train_total, test_total, df):
    errors_by_horizon = {horizon: [] for horizon in forecast_horizons}
    forecasts_by_horizon = {horizon: [] for horizon in forecast_horizons}

    for horizon in tqdm(forecast_horizons, desc='Horizon Progress'):
        if model_type == 'neural':
            if horizon != 1:
                default_config_AutoNBEATS=AutoNBEATS(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoNBEATS.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoNHITS=AutoNHITS(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoNHITS.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoDeepAR=AutoDeepAR(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoDeepAR.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoTFT=AutoTFT(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoTFT.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoMLP=AutoMLP(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoMLP.update({ 'early_stop_patience_steps':3, 'val_check_steps':10,})

                models = [
                    # AutoDeepAR(config = default_config_AutoDeepAR,  h=horizon,),#early_stop_patience_steps=1, val_check_steps=1
                    AutoTFT(config = default_config_AutoTFT, h=horizon,),
                    NBEATS(input_size=horizon*3, h=horizon, max_steps=500,  early_stop_patience_steps=3, val_check_steps=10),
                    NHITS(input_size=horizon*3, h=horizon, max_steps=500, early_stop_patience_steps=3, val_check_steps=10),
                    AutoMLP( config = default_config_AutoMLP, h=horizon,),
                    AutoNBEATS(config = default_config_AutoNBEATS, h=horizon,),
                    AutoNHITS(config = default_config_AutoNHITS, h=horizon,),
                ]
            elif horizon == 1:
                default_config_AutoNBEATS=AutoNBEATS(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoNBEATS.update({'stack_types':['identity'], 'early_stop_patience_steps':3, 'val_check_steps':10})

                default_config_AutoNHITS=AutoNHITS(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoNHITS.update({'stack_types':['identity'], 'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoDeepAR=AutoDeepAR(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoDeepAR.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoTFT=AutoTFT(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoTFT.update({'early_stop_patience_steps':3, 'val_check_steps':10,})

                default_config_AutoMLP=AutoMLP(h=horizon).get_default_config(h=horizon, backend='ray')
                default_config_AutoMLP.update({ 'early_stop_patience_steps':3, 'val_check_steps':10,})

                models = [
                    # AutoDeepAR(config = default_config_AutoDeepAR,  h=horizon,),#early_stop_patience_steps=1, val_check_steps=1
                    AutoTFT(config = default_config_AutoTFT, h=horizon,),
                    NBEATS(input_size=horizon*3, h=horizon, max_steps=500,  stack_types = ["identity"],early_stop_patience_steps=3, val_check_steps=10),
                    NHITS(input_size=horizon*3, h=horizon, max_steps=500,  stack_types = ["identity"],early_stop_patience_steps=3, val_check_steps=10),
                    AutoMLP( config = default_config_AutoMLP, h=horizon,),
                    AutoNBEATS(config = default_config_AutoNBEATS, h=horizon,),
                    AutoNHITS(config = default_config_AutoNHITS, h=horizon,),
                ]

        for model in tqdm(models, desc=f'Model Progress for Horizon {horizon}'):
            total_train_time = 0
            combined_forecasts = pd.DataFrame()
            combined_error = pd.DataFrame()

            
            log_to_file(f'Horizon: {horizon} for {model} has started running at {time.ctime()}')
            
            # Move the origin forward by one step at a time
            for start in range(0, test_size_total):
                if  train_size_total + start + horizon  > train_size_total + test_size_total:
                    break
                train_size = train_size_total + start
                train = df[:train_size]
                test = df[train_size:train_size + horizon]

                if model_type == 'statistical':
                    sf = StatsForecast(models=[model], freq='M', n_jobs=-1)
                    start_time = time.time()
                    forecasts_df = sf.forecast(df=train, h=horizon)

                elif model_type == 'tree':
                    fcst = MLForecast(
                        models=model[0],
                        freq="M",
                        target_transforms=[Differences([12]),Differences([24])],
                        lags=[1, 2, 3, 4, 11, 12, 18 , 24],
                        lag_transforms={1: [ExpandingMean()], 12: [RollingMean(window_size=12), rolling_mean_12], 24: [RollingMean(window_size=24), rolling_mean_24]},
                        date_features=[month_index]
                    )
                    model_tree = list(model[0])
                    start_time = time.time()
                    fcst.fit(train)
                    forecasts_df = fcst.predict(h=horizon)

                elif model_type == 'neural':
                    with contextlib.redirect_stdout(io.StringIO()):
                        nf = NeuralForecast(models=[model], freq='M')
                    start_time = time.time()
                    with contextlib.redirect_stdout(io.StringIO()):  # Suppress output
                        nf.fit(df=train, verbose=0, val_size=12)
                    with contextlib.redirect_stdout(io.StringIO()):
                        forecasts_df = nf.predict(verbose=False)
                    if model._get_name() == 'AutoDeepAR':
                        forecasts_df = forecasts_df[['unique_id' ,'ds', 'AutoDeepAR']]


                elif model_type == 'TimeGPT':
                    start_time = time.time()
                    if horizon > 12:
                        with contextlib.redirect_stdout(io.StringIO()):
                            forecasts_df = nixtla_client.forecast(df=train, h=horizon, freq='M', time_col='ds', target_col='y',model = 'timegpt-1-long-horizon')
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            forecasts_df = nixtla_client.forecast(df=train, h=horizon, freq='M', time_col='ds', target_col='y')
                    # print(forecasts_df)

                train_time = time.time() - start_time
                forecasts_df['origin'] = train_size  # Track the forecast origin point
                forecasts_df['horizon'] = horizon
                combined_forecasts = pd.concat([combined_forecasts, forecasts_df])

                error_df = calculate_errors(test, forecasts_df.drop(columns=['origin','horizon']), train)
                error_df['origin'] = train_size  # Track the forecast origin point
                error_df['horizon'] = horizon
                error_df['train_time'] = train_time

                combined_error = pd.concat([combined_error, error_df])

                total_train_time += train_time

            combined_forecasts = combined_forecasts.groupby('ds').mean(numeric_only=True).reset_index()
            combined_error['Total_Train_Time'] = total_train_time
            if model_type == 'tree':
                combined_forecasts['Model'] = model_tree[0]
                combined_error['Model'] = model_tree[0]
            else:
                combined_forecasts['Model'] = str(model)
                combined_error['Model'] = str(model)

            forecasts_by_horizon[horizon].append(combined_forecasts)
            errors_by_horizon[horizon].append(combined_error)


    return  errors_by_horizon, forecasts_by_horizon
# %%
# Forecast and evaluate neural models
neural_errors, neural_forecasts_by_horizon = forecast_and_evaluate_overlapping(neural_models_template, 'neural', forecast_horizons, train_total, test_total, df) # type: ignore


with open('neural_errors.pickle', 'wb') as handle:
    pickle.dump(neural_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('neural_forecasts_by_horizon.pickle', 'wb') as handle:  
    pickle.dump(neural_forecasts_by_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
with open('neural_forecasts_by_horizon.pickle', 'rb') as handle:
    neural_forecasts_by_horizon = pickle.load(handle)
    
    
with open('neural_errors.pickle', 'rb') as handle:
    neural_errors = pickle.load(handle)    


# %%
# Forecast and evaluate tree-based models
tree_errors , tree_forecasts_by_horizon  = forecast_and_evaluate_overlapping(tree_models, 'tree', forecast_horizons, train_total, test_total, df)
with open('tree_errors.pickle', 'wb') as handle:
    pickle.dump(tree_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tree_forecasts_by_horizon.pickle', 'wb') as handle:
    pickle.dump(tree_forecasts_by_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('tree_forecasts_by_horizon.pickle', 'rb') as handle:
    tree_forecasts_by_horizon = pickle.load(handle)
    
    
with open('tree_errors.pickle', 'rb') as handle:
    tree_errors = pickle.load(handle) 


# %%
# Forecast and evaluate statistical models
statistical_errors, statistical_forecasts_by_horizon  = forecast_and_evaluate_overlapping(statistical_models, 'statistical', forecast_horizons, train_total, test_total, df)
with open('statistical_errors.pickle', 'wb') as handle:
    pickle.dump(statistical_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('statistical_forecasts_by_horizon.pickle', 'wb') as handle:
    pickle.dump(statistical_forecasts_by_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('statistical_forecasts_by_horizon.pickle', 'rb') as handle:
    statistical_forecasts_by_horizon = pickle.load(handle)
    
    
with open('statistical_errors.pickle', 'rb') as handle:
    statistical_errors = pickle.load(handle)


# %%
TimeGPT_errors, TimeGPT_forecasts_by_horizon = forecast_and_evaluate_overlapping(['TimeGPT'], 'TimeGPT', forecast_horizons, train_total, test_total, df)
with open('TimeGPT_errors.pickle', 'wb') as handle:
    pickle.dump(TimeGPT_errors, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('TimeGPT_forecasts_by_horizon.pickle', 'wb') as handle:
    pickle.dump(TimeGPT_forecasts_by_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('TimeGPT_forecasts_by_horizon.pickle', 'rb') as handle:
    TimeGPT_forecasts_by_horizon = pickle.load(handle)
    
    
with open('TimeGPT_errors.pickle', 'rb') as handle:
    TimeGPT_errors = pickle.load(handle)


# %%
# import os

# output_dir = 'forecasts_by_horizon_five'
# os.makedirs(output_dir, exist_ok=True)

# def save_forecasts_by_horizon(forecasts_by_horizon, model_type):
#     for horizon, forecasts_list in forecasts_by_horizon.items():
#         combined_forecasts = pd.concat(forecasts_list).reset_index(drop=True)
#         combined_forecasts.to_csv(os.path.join(output_dir, f'{model_type}_forecasts_horizon_{horizon}.csv'), index=False)

# save_forecasts_by_horizon(statistical_forecasts_by_horizon, 'statistical')
# save_forecasts_by_horizon(tree_forecasts_by_horizon, 'tree')
# save_forecasts_by_horizon(TimeGPT_forecasts_by_horizon, 'TimeGPT')
# # save_forecasts_by_horizon(neural_forecasts_by_horizon, 'neural')


# %%
# %%
# Plot all forecasts combined
import plotly.graph_objects as go

def plot_all_forecasts(df, forecasts_by_horizon, model_type):
    fig = go.Figure()

    # Add actual data
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))

    # Add forecasts
    for horizon, forecasts_list in forecasts_by_horizon.items():
        combined_forecasts = pd.concat(forecasts_list).reset_index(drop=True)
        for col in combined_forecasts.columns:
            if col in ['ds', 'unique_id', 'origin','horizon','Model','index']:
                continue
            fig.add_trace(go.Scatter(x=combined_forecasts['ds'], y=combined_forecasts[col], mode='lines', name=f'{col} (Horizon {horizon})'))

    fig.update_layout(title=f'All Forecasts by Horizon ({model_type})', xaxis_title='Date', yaxis_title='Passengers')
    fig.show()

# Plot forecasts for statistical models
plot_all_forecasts(df, statistical_forecasts_by_horizon, 'statistical')

# Plot forecasts for tree-based models
plot_all_forecasts(df, tree_forecasts_by_horizon, 'tree')

# Plot forecasts for neural models
# plot_all_forecasts(df, neural_forecasts_by_horizon, 'neural')

# Plot forecasts for TimeGPT
plot_all_forecasts(df, TimeGPT_forecasts_by_horizon, 'TimeGPT')

# %%
# Plot forecasts separately by horizon
def plot_forecasts_by_horizon(df, forecasts_by_horizon, model_type):
    for horizon, forecasts_list in forecasts_by_horizon.items():
        fig = go.Figure()
        # Add actual data
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual'))

        # Add forecasts for the current horizon
        combined_forecasts = pd.concat(forecasts_list).reset_index(drop=True)
        for col in combined_forecasts.columns:
            if col in ['ds', 'unique_id', 'origin','horizon','Model','index']:
                continue
            fig.add_trace(go.Scatter(x=combined_forecasts['ds'], y=combined_forecasts[col], mode='lines', name=f'{col} (Horizon {horizon})'))

        fig.update_layout(title=f'Forecasts for Horizon {horizon} ({model_type})', xaxis_title='Date', yaxis_title='Passengers')
        fig.show()

# Plot forecasts for statistical models by horizon
plot_forecasts_by_horizon(df, statistical_forecasts_by_horizon, 'statistical')

# Plot forecasts for tree-based models by horizon
plot_forecasts_by_horizon(df, tree_forecasts_by_horizon, 'tree')

# Plot forecasts for neural models by horizon
plot_forecasts_by_horizon(df, neural_forecasts_by_horizon, 'neural')

# Plot forecasts for TimeGPT by horizon
plot_forecasts_by_horizon(df, TimeGPT_forecasts_by_horizon, 'TimeGPT')


# %%
# Collect and combine all errors from different horizons into a single DataFrame
def combine_errors(errors_by_horizon):
    combined_errors = []
    for horizon, errors_list in errors_by_horizon.items():
        for error_df in errors_list:
            error_df['Horizon'] = horizon
            combined_errors.append(error_df)
    combined_errors_df = pd.concat(combined_errors, ignore_index=True)
    return combined_errors_df

# all_errors_neural = combine_errors(neural_errors)
all_errors = pd.concat([combine_errors(tree_errors), combine_errors(statistical_errors),combine_errors(neural_errors), combine_errors(TimeGPT_errors)], ignore_index=True)
# Add other error combinations if needed


# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Find the best model for each horizon
all_errors_per_horizon = all_errors.groupby(['Horizon', 'Model']).agg({'RMSE': 'mean', 'SMAPE': 'mean', 'RMSSE': 'mean', 'train_time': 'sum'}).reset_index()
all_errors_per_horizon_std = all_errors.groupby(['Horizon', 'Model']).agg({'RMSE': 'std', 'SMAPE': 'std', 'RMSSE': 'std'}).reset_index()

# Merge mean and std dataframes
all_errors_per_horizon = pd.merge(all_errors_per_horizon, all_errors_per_horizon_std, on=['Horizon', 'Model'], suffixes=('', '_std'))


# %%
def create_plots(metric, horizon_range=None):
    # Filter data by horizon range if provided
    if horizon_range:
        data_filtered = all_errors_per_horizon[(all_errors_per_horizon['Horizon'] >= horizon_range[0]) & (all_errors_per_horizon['Horizon'] <= horizon_range[1])]
    else:
        data_filtered = all_errors_per_horizon

    # Best model for each horizon based on the chosen metric
    best_models = data_filtered.loc[data_filtered.groupby('Horizon')[metric].idxmin()]

    # Plot minimum errors by horizon
    fig_min_errors = px.line(best_models, x='Horizon', y=metric, title=f'Minimum {metric} by Horizon', markers=True, text='Model')
    fig_min_errors.update_traces(textposition='top center')

    # Performance stability (variance in the chosen metric for each model)
    performance_stability = data_filtered.groupby('Model')[metric].var().reset_index().sort_values(by=metric)

    fig_performance_stability = px.bar(performance_stability, x='Model', y=metric, title=f'Performance Stability (Variance in {metric})')

    # Training times
    total_train_times = data_filtered.groupby('Model')['train_time'].sum().reset_index().sort_values(by='train_time')
    fig_train_times = px.bar(total_train_times, x='Model', y='train_time', title='Total Training Times by Model')

    # Graphs per horizon sorted by their error with std added
    figs_per_horizon = []

    for horizon in data_filtered['Horizon'].unique():
        df_horizon = data_filtered[data_filtered['Horizon'] == horizon].sort_values(by=metric)
        fig_horizon = go.Figure()
        fig_horizon.add_trace(go.Bar(
            x=df_horizon['Model'],
            y=df_horizon[metric],
            name=f'Horizon {horizon}',
            error_y=dict(type='data', array=df_horizon[f'{metric}_std'])
        ))
        fig_horizon.update_layout(title=f'{metric} for Horizon {horizon}', xaxis_title='Model', yaxis_title=metric)
        figs_per_horizon.append(fig_horizon)

    # Box plot of errors using filtered all_errors
    fig_box_plot = px.box(all_errors[(all_errors['Horizon'] >= (horizon_range[0] if horizon_range else all_errors['Horizon'].min())) &
                                     (all_errors['Horizon'] <= (horizon_range[1] if horizon_range else all_errors['Horizon'].max()))],
                          x='Model', y=metric, color='Horizon', title=f'Box Plot of {metric} by Model and Horizon')

    # Show plots
    fig_min_errors.show()
    fig_performance_stability.show()
    fig_train_times.show()
    fig_box_plot.show()

    for fig_horizon in figs_per_horizon:
        fig_horizon.show()

def create_aggregated_plots(metric, short_horizon_range, medium_horizon_range, long_horizon_range):
    ranges = {
        'Short-term': short_horizon_range,
        'Medium-term': medium_horizon_range,
        'Long-term': long_horizon_range
    }
    
    for horizon_label, horizon_range in ranges.items():
        data_filtered = all_errors[(all_errors['Horizon'] >= horizon_range[0]) & (all_errors['Horizon'] <= horizon_range[1])]
        
        if data_filtered.empty:
            print(f"No data for {horizon_label} horizon range {horizon_range}")
            continue

        # Aggregate data
        data_agg = data_filtered.groupby('Model').agg({
            'RMSE': 'mean',
            'SMAPE': 'mean',
            'RMSSE': 'mean',
            'train_time': 'sum'
        }).reset_index()
        
        data_agg_std = data_filtered.groupby('Model').agg({
            'RMSE': 'std',
            'SMAPE': 'std',
            'RMSSE': 'std'
        }).reset_index()
        
        data_agg = pd.merge(data_agg, data_agg_std, on='Model', suffixes=('', '_std'))

        # Best model based on the chosen metric
        best_models = data_agg.loc[data_agg.groupby('Model')[metric].idxmin()]

        # Plot minimum errors by horizon
        fig_min_errors = px.line(best_models.sort_values(by=metric), x='Model', y=metric, title=f'Minimum {metric} by Model ({horizon_label} Horizon)', markers=True, text='Model')
        fig_min_errors.update_traces(textposition='top center')

        # # Performance stability (variance in the chosen metric for each model)
        # performance_stability = data_agg.groupby('Model')[metric].var().reset_index().sort_values(by=metric)

        # fig_performance_stability = px.bar(performance_stability, x='Model', y=metric, title=f'Performance Stability (Variance in {metric}) ({horizon_label} Horizon)')

        # Training times
        total_train_times = data_agg.groupby('Model')['train_time'].sum().reset_index().sort_values(by='train_time')
        fig_train_times = px.bar(total_train_times, x='Model', y='train_time', title=f'Total Training Times by Model ({horizon_label} Horizon)')

        # Box plot of errors
        fig_box_plot = px.box(data_filtered.sort_values(by=metric), x='Model', y=metric, title=f'Box Plot of {metric} by Model ({horizon_label} Horizon)')

        # Show plots
        fig_min_errors.show()
        fig_train_times.show()
        fig_box_plot.show()
        
        
def create_aggregated_plots_total(metric):
        # Best model for each horizon based on the chosen metric
    best_models = all_errors_per_horizon.loc[all_errors_per_horizon.groupby('Horizon')[metric].idxmin()]

    # Plot minimum errors by horizon
    fig_min_errors_horizon = px.line(best_models, x='Horizon', y=metric, title=f'Minimum {metric} by Horizon', markers=True, text='Model')
    fig_min_errors_horizon.update_traces(textposition='top center')
    # Aggregate data
    data_agg = all_errors.groupby('Model').agg({
        'RMSE': 'mean',
        'SMAPE': 'mean',
        'RMSSE': 'mean',
        'train_time': 'sum'
    }).reset_index()
    
    data_agg_std = all_errors.groupby('Model').agg({
        'RMSE': 'std',
        'SMAPE': 'std',
        'RMSSE': 'std'
    }).reset_index()
    
    data_agg = pd.merge(data_agg, data_agg_std, on='Model', suffixes=('', '_std'))

    # Best model based on the chosen metric
    best_models = data_agg.loc[data_agg.groupby('Model')[metric].idxmin()]

    # Plot minimum errors by horizon
    fig_min_errors = px.line(best_models.sort_values(by=metric), x='Model', y=metric, title=f'Minimum {metric} Total', markers=True, text='Model')
    fig_min_errors.update_traces(textposition='top center')

    # Training times
    total_train_times = data_agg.groupby('Model')['train_time'].sum().reset_index().sort_values(by='train_time')
    fig_train_times = px.bar(total_train_times, x='Model', y='train_time', title=f'Total Training Times by Model Total')

    # Box plot of errors
    fig_box_plot = px.box(all_errors.sort_values(by=metric), x='Model', y=metric, title=f'Box Plot of {metric} by Model Total')

    # Show plots
    fig_min_errors_horizon.show()
    fig_min_errors.show()
    fig_train_times.show()
    fig_box_plot.show()        

# Function to plot RMSE, SMAPE, or RMSSE
def plot_metric(metric, short_horizon_range=[1, 4], medium_horizon_range=[5, 12], long_horizon_range=[13, 18]):
    if metric not in ['RMSE', 'SMAPE', 'RMSSE']:
        print("Invalid metric! Choose from 'RMSE', 'SMAPE', or 'RMSSE'.")
        return
    
    # create_plots(metric)
    create_aggregated_plots_total(metric)
    create_aggregated_plots(metric, short_horizon_range, medium_horizon_range, long_horizon_range)

# Example usage: plot any one of the errors with specified horizon ranges
plot_metric('RMSE', short_horizon_range=[1, 4], medium_horizon_range=[5, 12], long_horizon_range=[13, 18])
# plot_metric('SMAPE', short_horizon_range=[1, 4], medium_horizon_range=[5, 12], long_horizon_range=[13, 18])
# plot_metric('RMSSE', short_horizon_range=[1, 4], medium_horizon_range=[5, 12], long_horizon_range=[13, 18])


