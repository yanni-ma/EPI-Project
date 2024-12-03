import pandas as pd

ccc_data = pd.read_csv('../datasets/ccc_preprocessed.csv')

# Filter rows with at least one valid size estimate
ccc_data = ccc_data.dropna(subset=['size_mean', 'size_low', 'size_high'], how='all')

# Save the filtered data to a new CSV file
ccc_data.to_csv('../datasets/ccc_filtered.csv', index=False)

print("Filtered data saved to '../datasets/ccc_filtered.csv'")
