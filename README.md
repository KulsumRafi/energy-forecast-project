🔋 Energy Consumption Forecasting Dashboard

📊 Project Overview

An interactive dashboard for forecasting building energy consumption using SARIMA (Seasonal ARIMA) time-series models. This project demonstrates end-to-end data analysis capabilities including time series modeling, interactive visualization, and business insight generation.

🎯 Key Features

- **Interactive Forecasts**: Adjust forecast horizon and confidence intervals in real-time
- **Anomaly Detection**: Automatically identifies unusual consumption patterns
- **Heat Map Visualization**: Reveals consumption patterns by hour and day
- **Model Diagnostics**: ACF/PACF plots and residual analysis
- **Business Impact Analysis**: Converts technical metrics to dollar values
- **Export Functionality**: Download forecasts as CSV files

🏗️ Project Structure
energy-forecast-project/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── model.pkl # Pre-trained SARIMA model
├── data/
│ ├── processed_data.csv # Cleaned time series data
│ └── feature_store.pkl # Feature store for quick access
├── .streamlit/
│ └── config.toml # Streamlit theme configuration
└── README.md # Project documentation


🔧 Technical Implementation

Data Processing
- Hourly smart meter readings aggregated to daily patterns
- Feature engineering: time-of-day, day-of-week, seasonal indicators
- Handling of missing values and outliers

Modeling Approach
- **SARIMA(p,d,q)(P,D,Q,s)** model with weekly seasonality (s=7)
- Automated parameter selection using ACF/PACF analysis
- Confidence intervals for uncertainty quantification
- Residual analysis for model validation

Model Performance
- **MAPE**: 5.8% on test set
- **RMSE**: 42.3 kWh (≈ $6.35 per day)
- **Forecast Bias**: -2.1% (slight under-forecast)

📈 Business Insights

The analysis revealed several actionable insights:

1. **Peak Demand Management**: Peak consumption occurs Tue-Thu, 3-6 PM
2. **Idle Consumption**: 15% of consumption occurs during unoccupied hours (11 PM - 5 AM)
3. **Anomaly Impact**: Detected anomalies account for approximately $450/month in excess costs
4. **Seasonal Patterns**: Summer months show 23% higher consumption vs spring

Dashboard Sections
1. Executive Dashboard
Real-time KPI cards

Current consumption vs forecast

Peak demand tracking

2. Forecast Analysis
Interactive forecast visualization

Confidence intervals

Historical vs predicted overlay

3. Pattern Analysis
Hour/day heat map

Weekly seasonality decomposition

Trend analysis

4. Anomaly Detection
Automated outlier identification

Impact assessment

Cause classification

5. Model Performance
Residual diagnostics

ACF/PACF plots

Accuracy metrics