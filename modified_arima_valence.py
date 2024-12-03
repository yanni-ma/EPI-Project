import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from data_preprocessing import load_nyt_data, load_ccc_data


nyt_filepath = './datasets/us-counties-2020.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

covid_data = load_nyt_data(nyt_filepath)
event_data = load_ccc_data(ccc_filepath)

merged_data = covid_data.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)

merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data = merged_data[merged_data['date'].dt.year == 2020]

merged_data = merged_data.sort_values('size_mean', ascending=False)
merged_data = merged_data.drop_duplicates(subset=['date', 'fips'], keep='first')

merged_data = merged_data.sort_values('date').reset_index(drop=True)

merged_data['has_event'] = merged_data['fips_code'].notna().astype(int)
merged_data['valence'] = merged_data['valence'].fillna(0)
merged_data['size_mean'] = merged_data['size_mean'].fillna(0)

county = '06037'  # LA County
county_data = merged_data[merged_data['fips'] == county]

if county_data.empty:
    print("No data available for the selected county or FIPS code.")
    exit()


county_data = county_data[['new_cases', 'has_event', 'valence', 'date']].dropna().reset_index(drop=True)

print(f"Number of unique dates in county_data: {county_data['date'].nunique()}")
print(f"Total rows in county_data: {len(county_data)}")

county_data['adjusted_cases'] = county_data['new_cases'].replace(0, pd.NA).interpolate(method='linear').fillna(0)

if county_data.empty:
    print("No valid data after cleaning.")
    exit()


split_point = int(len(county_data) * 0.75)
train_data = county_data.iloc[:split_point]
test_data = county_data.iloc[split_point:]

model = SARIMAX(
    train_data['adjusted_cases'],
    exog=train_data[['has_event', 'valence']],
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 7)
)
results = model.fit(maxiter=1000)

forecast_steps = len(test_data)
valence_scenarios = [
    ('valence 2', 2, 'orange'),
    ('valence 1', 1, 'green'),
    ('valence 0', 0, 'blue')
]

plt.figure(figsize=(15, 9))

plt.plot(train_data['adjusted_cases'], label='Training Data (Observed)', color='black')
plt.plot(
    range(split_point, split_point + forecast_steps),
    test_data['adjusted_cases'],
    label='Testing Data (Observed)',
    color='gray'
)

original_case_counts = covid_data[covid_data['fips'] == county]
if not original_case_counts.empty:
    plt.plot(
        range(len(original_case_counts)),
        original_case_counts['new_cases'],
        label='Original Case Counts (Unmerged)',
        color='purple',
        linestyle='-'
    )

for label, valence_value, color in valence_scenarios:
    future_exog = [[1, valence_value]] * forecast_steps  
    forecast = results.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    plt.plot(
        range(split_point, split_point + forecast_steps),
        forecast_values,
        label=f'Forecast ({label})',
        linestyle='--',
        color=color
    )

    plt.fill_between(
        range(split_point, split_point + forecast_steps),
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color=color,
        alpha=0.3
    )

plt.title(f'SARIMAX Model: Impact of Event Valence on Case Counts (County: {county})')
plt.xlabel('Time (Days)')
plt.ylabel('New Cases')
plt.legend()
plt.grid()
plt.savefig(f'./output/SARIMAX_forecast_valence_comparison_with_raw_county_{county}.png')
plt.show()
