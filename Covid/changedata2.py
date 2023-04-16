import pandas as pd

# Read the CSV file
df = pd.read_csv('Coviddata.csv')

# Combine the four columns with 0 and 1 as values
df['Age'] = df.apply(lambda row: ''.join(map(str, [row['Age_0-9'], row['Age_10-19'], row['Age_20-24'], row['Age_25-59'], row['Age_60+']])), axis=1)

# Replace the combined values with 1, 2, 3, and 4
df['Age'] = df['covid_range'].replace({'10000': 1, '01000': 2, '00100': 3, '00010': 4,'00001': 5})

# Save the modified CSV file
df.to_csv('Covid-data.csv', index=False)
