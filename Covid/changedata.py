import pandas as pd

# Read the CSV file
df = pd.read_csv('your_file.csv')

# Combine the four columns with 0 and 1 as values
df['combined'] = df.apply(lambda row: ''.join(map(str, [row['col1'], row['col2'], row['col3'], row['col4']])), axis=1)

# Replace the combined values with 1, 2, 3, and 4
df['combined'] = df['combined'].replace({'0000': 1, '0001': 2, '0010': 3, '0011': 4})

# Save the modified CSV file
df.to_csv('your_modified_file.csv', index=False)
