import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
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

merged_data_2020['date'] = pd.to_datetime(merged_data_2020['date'])
merged_data_2020 = merged_data_2020[merged_data_2020['date'].dt.year == 2020]
merged_data_2020 = merged_data_2020.sort_values('size_mean', ascending=False)
merged_data_2020 = merged_data_2020.drop_duplicates(subset=['date', 'fips'], keep='first')
merged_data_2020 = merged_data_2020.sort_values('date').reset_index(drop=True)
merged_data_2020['has_event'] = merged_data_2020['fips_code'].notna().astype(int)
merged_data_2020['valence'] = merged_data_2020['valence'].fillna(0)

## testing to make sure it's not just projecting more cases for orange since that's the positive one
## result of test: same result whether orange (right-leaning) or green (left-leaning) is positive
# merged_data_2020['valence'] = merged_data_2020['valence'].replace({1: -1, 2: 1})
merged_data_2020['valence'] = merged_data_2020['valence'].replace({1: 1, 2: -1})

merged_data_2020['size_mean'] = merged_data_2020['size_mean'].fillna(0)

# weighted valence?
# merged_data_2020['weighted_valence'] = merged_data_2020['valence'] * merged_data_2020['size_mean']

county = '06037'  # Los Angeles County, chosen because it has data for all valence levels
county_data_2020 = merged_data_2020[merged_data_2020['fips'] == county]

if county_data_2020.empty:
    print("No data available for the selected county or FIPS code.")
    exit()

county_data_2020 = county_data_2020[['new_cases', 'has_event', 'valence', 'date']].dropna().reset_index(drop=True)
county_data_2020['adjusted_cases'] = county_data_2020['new_cases'].replace(0, pd.NA).interpolate(method='linear').fillna(0)

county_data_2021 = covid_data_2021[covid_data_2021['fips'] == county]
county_data_2021 = county_data_2021[['date', 'new_cases']].dropna().reset_index(drop=True)
county_data_2021['adjusted_cases'] = county_data_2021['new_cases'].replace(0, pd.NA).interpolate(method='linear').fillna(0)

county_data_2021 = county_data_2021.iloc[:30].reset_index(drop=True)

# training SARIMAX model on 2020 data
model = SARIMAX(
    county_data_2020['adjusted_cases'],
    exog=county_data_2020[['has_event', 'valence']],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
results = model.fit(maxiter=1000)

forecast_steps = len(county_data_2021)
# valence_scenarios = [
#     ('valence 2', 1, 'orange'),  # rescaled to 1
#     ('valence 1', -1, 'green'),  # rescaled to -1
#     ('valence 0', 0, 'blue')     # neutral
# ]

## testing to make sure it's not just projecting more cases for orange since that's the positive one
valence_scenarios = [
    ('valence 2 (right-leaning events)', -1, 'orange'),  
    ('valence 1 (left-leaning events)', 1, 'green'),  
    ('valence 0 (neutral events)', 0, 'blue')    
]

plt.figure(figsize=(15, 9))

plt.plot(
    county_data_2020['adjusted_cases'],
    label='Training Data (2020)',
    color='black'
)

plt.plot(
    range(len(county_data_2020), len(county_data_2020) + forecast_steps),
    county_data_2021['adjusted_cases'],
    label='Ground Truth (First 30 Days of 2021)',
    color='grey'
)

for label, valence_value, color in valence_scenarios:
    future_exog = [[1, valence_value * merged_data_2020['size_mean'].mean()]] * forecast_steps
    forecast = results.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    plt.plot(
        range(len(county_data_2020), len(county_data_2020) + forecast_steps),
        forecast_values,
        label=f'Forecast ({label})',
        linestyle='--',
        color=color
    )

    # confidence intervals commented out for clarity
    # plt.fill_between(
    #     range(len(county_data_2020), len(county_data_2020) + forecast_steps),
    #     forecast_ci.iloc[:, 0],
    #     forecast_ci.iloc[:, 1],
    #     color=color,
    #     alpha=0.3
    # )

plt.title(f'SARIMAX Model: Valence Impact on Case Counts (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/final_SARIMAX_forecast__valence_comparison_2021_county_{county}.png')
plt.show()