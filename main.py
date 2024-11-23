import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from data_preprocessing import load_nyt_data, load_ccc_data

def detect_spikes(county_data, window=7, z_threshold=2):
    county_data['rolling_mean'] = county_data['new_cases'].rolling(window=window).mean()
    county_data['rolling_std'] = county_data['new_cases'].rolling(window=window).std()
    county_data['z_score'] = (county_data['new_cases'] - county_data['rolling_mean']) / county_data['rolling_std']
    county_data['is_spike'] = county_data['z_score'] > z_threshold

    clean_data = county_data.dropna(subset=['rolling_mean', 'rolling_std']).copy()

    height_threshold = clean_data['rolling_mean'] + z_threshold * clean_data['rolling_std']

    peaks, _ = find_peaks(clean_data['new_cases'], height=height_threshold.values)
    county_data['is_peak'] = False
    county_data.loc[clean_data.iloc[peaks].index, 'is_peak'] = True

    return county_data

def correlate_spikes_with_events(county_data, events, time_window=14):
    correlations = []
    for _, event in events.iterrows():
        event_date = event['date']
        start_date = event_date - pd.Timedelta(days=time_window)
        end_date = event_date + pd.Timedelta(days=time_window)
        spike_cases = county_data.loc[start_date:end_date, 'new_cases']
        correlations.append({
            'event_date': event_date,
            'event_type': event.get('event_type', 'Unknown'),
            'event_valence': event.get('event_valence', 'Neutral'),
            'event_size': event.get('event_size', 'N/A'),
            'mean_cases_near_event': spike_cases.mean(),
            'max_cases_near_event': spike_cases.max(),
            'total_cases_near_event': spike_cases.sum()
        })
    return pd.DataFrame(correlations)

def plot_spikes(county_data):
    plt.figure(figsize=(12, 6))
    plt.plot(county_data['new_cases'], label="New Cases", color="blue")
    plt.scatter(county_data.index[county_data['is_spike']],
                county_data['new_cases'][county_data['is_spike']],
                color="red", label="Detected Spikes")
    plt.title("COVID-19 Case Counts with Detected Spikes")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.legend()
    plt.grid()
    plt.show()

def plot_events_vs_spikes(county_data, events):
    plt.figure(figsize=(12, 6))
    plt.plot(county_data['new_cases'], label="New Cases", color="blue")
    plt.scatter(county_data.index[county_data['is_spike']],
                county_data['new_cases'][county_data['is_spike']],
                color="red", label="Detected Spikes")
    for event_date in events['date']:
        plt.axvline(x=event_date, color='orange', linestyle='--', label='Event Date')
    plt.title("COVID-19 Case Counts with Events and Detected Spikes")
    plt.xlabel("Date")
    plt.ylabel("New Cases")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_valence_effects(correlation_results):
    valence_groups = correlation_results.groupby('event_valence')
    valence_summary = valence_groups['max_cases_near_event'].mean().sort_values(ascending=False)
    print("Average Max Cases Near Events by Valence:")
    print(valence_summary)
    return valence_summary

def main():
    nyt_filepath = './datasets/us-counties-2020.csv'
    ccc_filepath = './datasets/ccc_filtered.csv'

    print("Loading datasets...")
    nyt_data = load_nyt_data(nyt_filepath)
    ccc_data = load_ccc_data(ccc_filepath)

    if nyt_data.empty:
        raise ValueError("NYT dataset is empty. Check the file and preprocessing.")
    if ccc_data.empty:
        raise ValueError("CCC dataset is empty. Check the file and preprocessing.")

    county_fips = '53061'
    print(f"Filtering data for county FIPS: {county_fips}")

    county_data = nyt_data[nyt_data['fips'] == county_fips].set_index('date')
    if county_data.empty:
        raise ValueError(f"No data found for FIPS {county_fips} in NYT dataset.")

    county_data['new_cases'] = county_data['new_cases'].clip(lower=0)

    print("Detecting spikes in case data...")
    county_data = detect_spikes(county_data)

    plot_spikes(county_data)

    print("Analyzing correlation with events...")
    county_events = ccc_data[(ccc_data['fips_code'] == county_fips) & (ccc_data['date'] >= '2020-01-01')]
    if not county_events.empty:
        correlation_results = correlate_spikes_with_events(county_data, county_events)
        print("Correlation results:")
        print(correlation_results)
        plot_events_vs_spikes(county_data, county_events)
        analyze_valence_effects(correlation_results)
    else:
        print("No CCC events found for the selected county.")

if __name__ == "__main__":
    main()
