import pandas as pd

# Read the CSV file into a DataFrame
data_frame = pd.read_csv('LoanTrain.csv')

# Convert DataFrame to a NumPy array
data_array = data_frame.values
