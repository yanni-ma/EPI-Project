import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from data_preprocessing import load_nyt_data, load_ccc_data

nyt_filepath_2020 = './datasets/us-counties-2020.csv'
ccc_filepath = './datasets/ccc_filtered.csv'

covid_data_2020 = load_nyt_data(nyt_filepath_2020)
event_data = load_ccc_data(ccc_filepath)

merged_data_2020 = covid_data_2020.merge(
    event_data,
    left_on=['date', 'fips'],
    right_on=['date', 'fips_code'],
    how='left'
)


ccc_data_2020 = event_data[event_data['date'].dt.year == 2020]
event_counts = ccc_data_2020.groupby('fips_code').size()

with open("./datasets/county_list.txt") as file:
    for county in file:
        county = county.strip()
        county_data = merged_data_2020[merged_data_2020['fips'] == county].copy()

        county_data['new_cases'] = county_data['cases'].diff().fillna(0).clip(lower=0)
        population = 844761

        def seir_model(y, t, N, beta, sigma, gamma):
            S, E, I, R = y
            dSdt = -beta * S * I / N
            dEdt = beta * S * I / N - sigma * E
            dIdt = sigma * E - gamma * I
            dRdt = gamma * I
            return dSdt, dEdt, dIdt, dRdt

        num_events = event_counts.get(county, 0)
        if num_events == 0:
            beta = 0.115
        else:
            beta = 0.115 * np.log(num_events) # measure against how many events
        sigma = 1/5.2 #5.2 day incubation period
        gamma = 1/10 #10 days to recover
        try:
            I0 = county_data['new_cases'].iloc[0]
            i = 1
            while I0 == 0:
                I0 = county_data['new_cases'].iloc[i]
                i += 1
        except:
            continue
        E0 = I0 
        R0 = 0 
        S0 = population - I0 - E0 - R0  
        N = population

        y0 = S0, E0, I0, R0

        t = np.linspace(0, len(county_data), len(county_data))

        ret = odeint(seir_model, y0, t, args=(N, beta, sigma, gamma))
        S, E, I, R = ret.T

        plt.figure(figsize=(14, 7))
        plt.plot(t, S, label='Susceptible', color='blue')
        plt.plot(t, E, label='Exposed', color='orange')
        plt.plot(t, I, label='Infectious', color='red')
        plt.plot(t, R, label='Recovered', color='green')
        plt.scatter(range(len(county_data)), county_data['new_cases'], label='Observed Cases', color='purple', alpha=0.6)
        plt.ylim(0, county_data['new_cases'].max() * 1.2)
        plt.title(f'SEIR Model for County {county}')
        plt.xlabel('Time (Days)')
        plt.ylabel('Population')
        plt.legend()
        plt.grid()
        plt.savefig(f'./output/SEIR_model_county_{county}.png')
        plt.show()
