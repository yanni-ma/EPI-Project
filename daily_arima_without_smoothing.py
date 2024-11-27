## code is copied from daily_arima.py, this is to show what it looks like without smoothing
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from data_preprocessing import load_nyt_data, load_ccc_data

nyt_filepath = './datasets/us-counties-2020.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

covid_data = load_nyt_data(nyt_filepath)
event_data = load_ccc_data(ccc_filepath)

# merge COVID data with event data
merged_data = covid_data.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)

# if missing event-related columns, fill with default values
merged_data['has_event'] = merged_data['fips_code'].notna().astype(int)
merged_data['valence'] = merged_data['valence'].fillna(0)
merged_data['size_mean'] = merged_data['size_mean'].fillna(0)

county = '53061'
county_data = merged_data[merged_data['fips'] == county]


if county_data.empty:
    print("no data available for the selected county or FIPS code.")
    exit()

# drop rows with NaN in relevant columns
county_data = county_data[['new_cases', 'has_event']].dropna().reset_index(drop=True)

county_data['adjusted_cases'] = county_data['new_cases'].replace(0, pd.NA).interpolate(method='linear').fillna(0)
# forward fill to handle zeros in the data 
county_data['adjusted_cases'] = county_data['new_cases'].replace(0, pd.NA).fillna(method='ffill').fillna(0)

if county_data.empty:
    print("no valid data after cleaning")
    exit()

# fit SARIMAX model
model = SARIMAX(
    county_data['new_cases'], # uncomment this line to use original data, not adjusted cases (smoothed data)
    exog=county_data[['has_event']],
    order=(1, 1, 1),  # can adjust order params 
    seasonal_order=(1, 0, 1, 7)  # can adjust seasonal params
)
results = model.fit()

forecast = results.get_forecast(steps=30, exog=[[1]] * 30)  # can adjust these 
forecast_ci = forecast.conf_int()


plt.figure(figsize=(15, 9))

# observed cases
plt.plot(county_data['new_cases'], label='Observed New Cases', color='blue')

# forecasted cases
plt.plot(
    range(len(county_data), len(county_data) + 30),
    forecast.predicted_mean,
    label='Forecast',
    color='orange'
)

# show confidence intervals
plt.fill_between(
    range(len(county_data), len(county_data) + 30),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color='orange',
    alpha=0.3
)

plt.title('SARIMAX Model: Observed and Forecasted New Cases for county: ' + county)
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig('./output/no_smoothing_SARIMAX_forecast_county_' + county + '.png')
plt.show()