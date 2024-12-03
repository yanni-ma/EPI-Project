import pandas as pd

def load_nyt_data(filepath):
    nyt_data = pd.read_csv(filepath, parse_dates=['date'])
    
    # drops rows with missing FIPS codes
    nyt_data = nyt_data.dropna(subset=['fips'])
    
    # format FIPS
    nyt_data['fips'] = nyt_data['fips'].astype(str).str.split('.').str[0].str.zfill(5)
    
    # sort by FIPS and date
    nyt_data = nyt_data.sort_values(['fips', 'date'])
    
    # calculate daily new cases
    nyt_data['new_cases'] = nyt_data.groupby('fips')['cases'].diff().fillna(0)
    
    return nyt_data

def load_ccc_data(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
        ccc_data = pd.read_csv(file, parse_dates=['date'])
    
    ccc_data['fips_code'] = ccc_data['fips_code'].astype(str).str.split('.').str[0].str.zfill(5)
    
    # drop events before 2020
    ccc_data = ccc_data[ccc_data['date'] >= '2020-01-01']
    
    return ccc_data