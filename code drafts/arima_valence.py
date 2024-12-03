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

merged_data['has_event'] = merged_data['fips_code'].notna().astype(int)
merged_data['valence'] = merged_data['valence'].fillna(0)
merged_data['size_mean'] = merged_data['size_mean'].fillna(0)

with open("./datasets/county_list.txt") as file:
    for county in file:
        county = county.strip()
        county_data = merged_data[merged_data['fips'] == county]

        if county_data.empty:
            print("No data available for the selected county or FIPS code.")
            continue

        county_data = county_data[['new_cases', 'valence']].dropna().reset_index(drop=True)

        county_data['adjusted_cases'] = county_data['new_cases'].replace(0, pd.NA).fillna(method='ffill').fillna(0)

        print("county")
        print(county_data)
        if county_data.empty:
            print("No valid data after cleaning.")
            exit()

        valence_positive = county_data[county_data['valence'] > 0].reset_index(drop=True)
        valence_negative = county_data[county_data['valence'] == 0].reset_index(drop=True)

        def fit_and_forecast(data, valence_label):
            if len(data) < 30:
                print(f"No data available for {valence_label}.")
                return None, None, None
            
            model = SARIMAX(
                data['adjusted_cases'],
                exog=pd.DataFrame(data['valence']),
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7)
            )
            results = model.fit(maxiter=1000)
            forecast = results.get_forecast(steps=30, exog=[[1]] * 30)
            forecast_ci = forecast.conf_int()
            return results, forecast, forecast_ci

        results_positive, forecast_positive, ci_positive = fit_and_forecast(valence_positive, 'valence > 0')
        results_negative, forecast_negative, ci_negative = fit_and_forecast(valence_negative, 'valence == 0')

        if not (results_negative and results_positive):
            continue

        plt.figure(figsize=(14, 7))

        if results_positive:
            plt.plot(
                range(len(valence_positive)),
                valence_positive['adjusted_cases'],
                label='Observed (Valence > 0)',
                color='blue'
            )
            plt.plot(
                range(len(valence_positive), len(valence_positive) + 30),
                forecast_positive.predicted_mean,
                label='Forecast (Valence > 0)',
                color='orange'
            )
            plt.fill_between(
                range(len(valence_positive), len(valence_positive) + 30),
                ci_positive.iloc[:, 0],
                ci_positive.iloc[:, 1],
                color='orange',
                alpha=0.3
            )

        if results_negative:
            plt.plot(
                range(len(valence_negative)),
                valence_negative['adjusted_cases'],
                label='Observed (Valence == 0)',
                color='green'
            )
            plt.plot(
                range(len(valence_negative), len(valence_negative) + 30),
                forecast_negative.predicted_mean,
                label='Forecast (Valence == 0)',
                color='red'
            )
            plt.fill_between(
                range(len(valence_negative), len(valence_negative) + 30),
                ci_negative.iloc[:, 0],
                ci_negative.iloc[:, 1],
                color='red',
                alpha=0.3
            )

        plt.title(f'SARIMAX Model: Comparison of Valence > 0 and Valence == 0 for county {county}')
        plt.xlabel('Time (Days)')
        plt.ylabel('New Cases')
        plt.legend()
        plt.grid()
        plt.savefig('./output/SARIMAX_forecast_comparison_county_' + county + '.png')
