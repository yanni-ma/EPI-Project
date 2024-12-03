import pandas as pd

with open('../datasets/ccc_compiled_2017-2020.csv', 'r', encoding='utf-8', errors='replace') as file:
    ccc_data = pd.read_csv(file)

columns_to_keep = ['date', 'fips_code', 'valence', 'size_mean', 'size_low', 'size_high', 'type']
ccc_data = ccc_data[columns_to_keep]

ccc_data['fips_code'] = ccc_data['fips_code'].astype(str).str.zfill(5)

ccc_data.to_csv('../datasets/ccc_preprocessed.csv', index=False)

print("Preprocessed data saved to '../datasets/ccc_preprocessed.csv'")
