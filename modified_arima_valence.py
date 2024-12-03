import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from data_preprocessing import load_nyt_data, load_ccc_data

nyt_filepath = './datasets/us-counties-2020.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

# Load and merge datasets
covid_data = load_nyt_data(nyt_filepath)
event_data = load_ccc_data(ccc_filepath)

merged_data = covid_data.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)

# Fill missing values for event-related columns
merged_data['has_event'] = merged_data['fips_code'].notna().astype(int)
merged_data['valence'] = merged_data['valence'].fillna(0)
merged_data['size_mean'] = merged_data['size_mean'].fillna(0)

with open("./datasets/county_list.txt") as file:
    for county in file:
        county = county.strip()
        county_data = merged_data[merged_data['fips'] == county]

        if county_data.empty:
            print(f"No data available for the selected county {county}.")
            continue

        # Drop rows with NaN in relevant columns
        county_data = county_data[['new_cases', 'valence']].dropna().reset_index(drop=True)

        county_data['adjusted_cases'] = county_data['new_cases'].replace(0, pd.NA).fillna(method='ffill').fillna(0)

        if county_data.empty:
            print(f"No valid data after cleaning for county {county}.")
            continue

        # Filter data for each valence category
        valence_2 = county_data[county_data['valence'] == 2].reset_index(drop=True)
        valence_1 = county_data[county_data['valence'] == 1].reset_index(drop=True)
        valence_0 = county_data[county_data['valence'] == 0].reset_index(drop=True)

        # Function to fit SARIMAX model and forecast
        def fit_and_forecast(data, valence_label):
            if len(data) < 30:
                print(f"Not enough data for {valence_label}.")
                return None, None, None

            model = SARIMAX(
                data['adjusted_cases'],
                exog=pd.DataFrame(data['valence']),
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7)
            )
            results = model.fit(maxiter=1000)
            forecast = results.get_forecast(steps=30, exog=[[2 if valence_label == 'valence 2' else 1]] * 30)
            forecast_ci = forecast.conf_int()
            return results, forecast, forecast_ci

        # Fit models and generate forecasts
        results_2, forecast_2, ci_2 = fit_and_forecast(valence_2, 'valence 2')
        results_1, forecast_1, ci_1 = fit_and_forecast(valence_1, 'valence 1')
        results_0, forecast_0, ci_0 = fit_and_forecast(valence_0, 'valence 0')

        # Plot comparison
        plt.figure(figsize=(14, 7))

        for label, results, forecast, ci, color in [
            ('valence 2', results_2, forecast_2, ci_2, 'orange'),
            ('valence 1', results_1, forecast_1, ci_1, 'green'),
            ('valence 0', results_0, forecast_0, ci_0, 'blue')
        ]:
            if results:
                # Observed cases
                plt.plot(
                    range(len(county_data)),
                    county_data['adjusted_cases'],
                    label=f'Observed ({label})',
                    color=color
                )
                # Forecasted cases
                plt.plot(
                    range(len(county_data), len(county_data) + 30),
                    forecast.predicted_mean,
                    label=f'Forecast ({label})',
                    linestyle='--',
                    color=color
                )
                # Confidence intervals
                plt.fill_between(
                    range(len(county_data), len(county_data) + 30),
                    ci.iloc[:, 0],
                    ci.iloc[:, 1],
                    color=color,
                    alpha=0.3
                )

        plt.title(f'SARIMAX Model: Impact of Event Valence on Case Counts (County: {county})')
        plt.xlabel('Time (Days)')
        plt.ylabel('New Cases')
        plt.legend()
        plt.grid()
        plt.savefig(f'./output/modified_SARIMAX_forecast_comparison_valence_county_{county}.png')
        plt.show()
