# Adding columns age and gender to data_complete.csv with default n/a string as values

# Importing required libraries
import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/data_complete.csv')

# Add the columns
df['age'] = 'n/a'
df['gender'] = 'n/a'

# Save the updated DataFrame to a new CSV file
df.to_csv('../data/data_complete.csv', index=False)