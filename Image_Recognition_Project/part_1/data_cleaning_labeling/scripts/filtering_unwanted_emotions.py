import pandas as pd

# Read the CSV file
df = pd.read_csv('../../fer2013_dataset/dataset.csv')

# Display the first few rows of the dataframe to verify contents
print("Original DataFrame:")
print(df.head())

# Filter out rows with emotions 1, 2, 4, and 5
filtered_df = df[~df['emotion'].isin([1, 2, 4, 5])]

# Display the first few rows of the filtered dataframe to verify filtering
print("Filtered DataFrame:")
print(filtered_df.head())

# Save the filtered data to a new CSV file
filtered_df.to_csv('../data/data_v1_filtered_emotions.csv', index=False)

print("Filtered data saved to 'data_v1_filtered_emotions.csv'")