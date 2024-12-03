import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_nyt_data, load_ccc_data

nyt_filepath_2020 = './datasets/us-counties-2020.csv'
nyt_filepath_2021 = './datasets/us-counties-2021.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

covid_data_2020 = load_nyt_data(nyt_filepath_2020)
covid_data_2021 = load_nyt_data(nyt_filepath_2021)
event_data = load_ccc_data(ccc_filepath)

merged_data_2020 = covid_data_2020.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)

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

model_smoothed = SARIMAX(
    county_data_2020['adjusted_cases'],
    exog=county_data_2020[['has_event']],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
results_smoothed = model_smoothed.fit(maxiter=1000)

model_unsmoothed = SARIMAX(
    county_data_2020['new_cases'],
    exog=county_data_2020[['has_event']],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
results_unsmoothed = model_unsmoothed.fit(maxiter=1000)

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

forecast_smoothed = results_smoothed.get_forecast(steps=len(county_data_2021_subset), exog=[[1]] * len(county_data_2021_subset))
forecast_unsmoothed = results_unsmoothed.get_forecast(steps=len(county_data_2021_subset), exog=[[1]] * len(county_data_2021_subset))

forecast_values_smoothed = forecast_smoothed.predicted_mean
forecast_values_unsmoothed = forecast_unsmoothed.predicted_mean

rmse_smoothed = mean_squared_error(county_data_2021_subset['adjusted_cases'], forecast_values_smoothed, squared=False)
rmse_unsmoothed = mean_squared_error(county_data_2021_subset['new_cases'], forecast_values_unsmoothed, squared=False)

print(f"RMSE (Smoothed): {rmse_smoothed}")
print(f"RMSE (Unsmoothed): {rmse_unsmoothed}")

plt.figure(figsize=(15, 9))
plt.plot(county_data_2020['adjusted_cases'], label='2020 Adjusted Cases (Smoothed)', color='blue')
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    county_data_2021_subset['adjusted_cases'],
    label='2021 Adjusted Cases (Observed - Subset)',
    color='green',
    linestyle='--'
)
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    forecast_values_smoothed,
    label='Forecast (Smoothed Data)',
    color='orange'
)
plt.title(f'ARIMA Smoothed Forecast vs 2021 Subset (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/ARIMA_forecast_smoothed_comparison_county_{county}.png')
plt.show()

plt.figure(figsize=(15, 9))
plt.plot(county_data_2020['new_cases'], label='2020 Observed Cases (Unsmoothed)', color='blue')
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    county_data_2021_subset['new_cases'],
    label='2021 Observed Cases (Unsmoothed - Subset)',
    color='green',
    linestyle='--'
)
plt.plot(
    range(len(county_data_2020), len(county_data_2020) + subset_length),
    forecast_values_unsmoothed,
    label='Forecast (Unsmoothed Data)',
    color='red'
)
plt.title(f'ARIMA Unsmoothed Forecast vs 2021 Subset (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/ARIMA_forecast_unsmoothed_comparison_county_{county}.png')
plt.show()
