# Multi-Origin Time Series Forecasting with Foundation Models

## Overview
This project implements a comprehensive time series forecasting framework utilizing a multi-origin approach to evaluate foundation models against traditional forecasting methods. We compare the performance of foundation models against 20 established neural networks, machine learning, and statistical benchmarks using rolling time origins and robust error metrics.

## Key Features
- Multi-origin evaluation framework
- Implementation of 20+ forecasting models including:
  - Foundation Models (TimeGPT, Chronos, Moirai)
  - Neural Networks (MLP, NBEATS, NHITS, etc.)
  - Statistical Models (ARIMA, ETS, etc.)
  - Machine Learning Models (XGBoost, LightGBM, etc.)
- Robust error metrics (RMSSE, SMAPE, etc.)
- Comprehensive visualization tools
## Results
Our findings demonstrate that Foundation models can outperform simple benchmarks and most neural networks in forecasting tasks, though they have not yet surpassed the performance of most statistical and machine-learning approaches.

## Usage: 
The main implementation is in `Final_model_accidental_Deaths.ipynb` script which contain the entire code to replicate the sure, you just need to install the required libraries and support for Pytorch with CUDA
