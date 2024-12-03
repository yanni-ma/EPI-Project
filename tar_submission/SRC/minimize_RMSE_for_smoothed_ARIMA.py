import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from itertools import product
from data_preprocessing import load_nyt_data, load_ccc_data
import matplotlib.pyplot as plt

nyt_filepath_2020 = './datasets/us-counties-2020.csv'
nyt_filepath_2021 = './datasets/us-counties-2021.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

# Load data
covid_data_2020 = load_nyt_data(nyt_filepath_2020)
covid_data_2021 = load_nyt_data(nyt_filepath_2021)
event_data = load_ccc_data(ccc_filepath)

# Merge COVID and event data for 2020
merged_data_2020 = covid_data_2020.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)

# Add event-related variables
merged_data_2020['has_event'] = merged_data_2020['fips_code'].notna().astype(int)
merged_data_2020['valence'] = merged_data_2020['valence'].fillna(0)
merged_data_2020['size_mean'] = merged_data_2020['size_mean'].fillna(0)

county = '53061'
county_data_2020 = merged_data_2020[merged_data_2020['fips'] == county]

if county_data_2020.empty:
    print("No data available for the selected county or FIPS code.")
    exit()

county_data_2020 = county_data_2020[['new_cases', 'has_event']].dropna().reset_index(drop=True)

county_data_2020['adjusted_cases'] = (
    county_data_2020['new_cases'].replace(0, pd.NA)
    .interpolate(method='linear')
    .fillna(method='ffill')
    .fillna(0)
)

county_data_2021 = covid_data_2021[covid_data_2021['fips'] == county]
county_data_2021 = county_data_2021[['date', 'cases']].dropna().reset_index(drop=True)
county_data_2021['new_cases'] = county_data_2021['cases'].diff().fillna(0)
county_data_2021['adjusted_cases'] = (
    county_data_2021['new_cases'].replace(0, pd.NA)
    .interpolate(method='linear')
    .fillna(method='ffill')
    .fillna(0)
)

subset_length = 60
county_data_2021_subset = county_data_2021[:subset_length]

p = d = q = range(0, 2)  
P = D = Q = range(0, 2)  
s = [7]

parameter_combinations = list(product(p, d, q, P, D, Q, s))

# Grid search
best_rmse = float('inf')
best_params = None

for (p, d, q, P, D, Q, s) in parameter_combinations:
    try:
        model = SARIMAX(
            county_data_2020['adjusted_cases'],
            exog=county_data_2020[['has_event']],  # Include exogenous variable
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(maxiter=1000, disp=False)
        forecast = results.get_forecast(steps=len(county_data_2021_subset), exog=[[1]] * len(county_data_2021_subset))
        forecast_values = forecast.predicted_mean
        rmse = mean_squared_error(county_data_2021_subset['adjusted_cases'], forecast_values, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (p, d, q, P, D, Q, s)
    except Exception as e:
        continue

print(f"Best RMSE: {best_rmse}")
print(f"Best Parameters: {best_params}")

# Fit model with best parameters
(p, d, q, P, D, Q, s) = best_params
best_model = SARIMAX(
    county_data_2020['adjusted_cases'],
    exog=county_data_2020[['has_event']],  # Include exogenous variable
    order=(p, d, q),
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
)
best_results = best_model.fit(maxiter=1000, disp=False)

# Generate forecast for first 60 days of 2021
forecast_best = best_results.get_forecast(steps=len(county_data_2021_subset), exog=[[1]] * len(county_data_2021_subset))
forecast_values_best = forecast_best.predicted_mean
forecast_ci = forecast_best.conf_int()

# Plot observed and forecasted data
plt.figure(figsize=(15, 9))

# Plot 2020 observed data
plt.plot(county_data_2020['adjusted_cases'], label='2020 Adjusted Cases (Smoothed)', color='blue')

# Plot 2021 observed data (first 60 days)
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    county_data_2021_subset['adjusted_cases'],
    label='2021 Adjusted Cases (Observed - Subset)',
    color='green',
    linestyle='--'
)

# Plot forecasted data
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    forecast_values_best,
    label='Forecast (Best Parameters with Exog)',
    color='orange'
)

# Plot confidence intervals
plt.fill_between(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='orange',
    alpha=0.3
)

plt.title(f'SARIMAX Forecast with Best Parameters (with Exog) vs 2021 Subset (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/Best_SARIMAX_forecast_with_exog_county_{county}.png')
plt.show()
