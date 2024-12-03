import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_nyt_data, load_ccc_data

nyt_filepath_2020 = './datasets/us-counties-2020.csv'
nyt_filepath_2021 = './datasets/us-counties-2021.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

covid_data_2020 = load_nyt_data(nyt_filepath_2020)
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
county_data = merged_data_2020[merged_data_2020['fips'] == county]

if county_data.empty:
    print("No data available for the selected county or FIPS code.")
    exit()

data_valence_0 = county_data[county_data['valence'] == 0]
data_valence_positive = county_data[county_data['valence'] > 0]

def process_arima(data, label):
    data = data[['new_cases', 'has_event']].dropna().reset_index(drop=True)
    data['adjusted_cases'] = (
        data['new_cases'].replace(0, pd.NA)
        .interpolate(method='linear')
        .fillna(method='ffill')
        .fillna(0)
    )
    model = SARIMAX(
        data['adjusted_cases'],
        exog=data[['has_event']],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7)
    )
    results = model.fit(maxiter=1000)
    return results, data

results_valence_0, data_valence_0 = process_arima(data_valence_0, 'Valence 0')
results_valence_positive, data_valence_positive = process_arima(data_valence_positive, 'Valence > 0')

forecast_steps = 60

future_exog_valence_0 = [[0]] * forecast_steps
future_exog_valence_positive = [[1]] * forecast_steps

forecast_valence_0 = results_valence_0.get_forecast(steps=forecast_steps, exog=future_exog_valence_0)
forecast_valence_positive = results_valence_positive.get_forecast(steps=forecast_steps, exog=future_exog_valence_positive)

forecast_values_valence_0 = forecast_valence_0.predicted_mean
forecast_values_valence_positive = forecast_valence_positive.predicted_mean

plt.figure(figsize=(15, 9))
plt.plot(
    data_valence_0['adjusted_cases'], label='Adjusted Cases (Valence 0)', color='blue'
)
plt.plot(
    forecast_values_valence_0, label='Forecast (Valence 0)', color='orange', linestyle='--'
)
plt.plot(
    data_valence_positive['adjusted_cases'], label='Adjusted Cases (Valence > 0)', color='green'
)
plt.plot(
    forecast_values_valence_positive, label='Forecast (Valence > 0)', color='red', linestyle='--'
)
plt.title(f'ARIMA Forecast by Valence (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/ARIMA_forecast_valence_comparison_county_{county}.png')
plt.show()
