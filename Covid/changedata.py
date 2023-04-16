import pandas as pd

# Read the CSV file
df = pd.read_csv('Cleaned-Data.csv')

# Combine the four columns with 0 and 1 as values
df['covid_range'] = df.apply(lambda row: ''.join(map(str, [row['Severity_None'], row['Severity_Mild'], row['Severity_Moderate'], row['Severity_Severe']])), axis=1)

# Replace the combined values with 1, 2, 3, and 4
df['covid_range'] = df['covid_range'].replace({'1000': 1, '0100': 2, '0010': 3, '0001': 4})

# Save the modified CSV file
df.to_csv('Coviddata.csv', index=False)
