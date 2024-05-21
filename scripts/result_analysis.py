import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file into a pandas DataFrame
# Replace 'your_file.csv' with the path to your actual CSV file
file_path = './concatenated_dataframe.csv'
df = pd.read_csv(file_path)

# Print the length of the DataFrame
print(f"Number of rows in the dataset: {len(df)}")

# Print the columns of the DataFrame
print("Columns in the dataset:")
print(df.columns)

# Print the first 5 rows of the DataFrame
print("First 5 rows of the dataset:")
print(df.head())

# Print some basic statistics about the DataFrame
print("Basic statistics of the dataset:")
print(df.describe())


unique_step_numbers = df['Step number'].unique()
unique_mae_loss = df['Generator mae loss'].unique()
print(unique_step_numbers[-100:])
print(unique_mae_loss[-100:])

column_to_plot = 'Generator mae loss'
plt.figure(figsize=(10, 6))
plt.plot(df[column_to_plot])
plt.title(f'Line Plot of {column_to_plot}')
plt.xlabel('Index')
plt.ylabel(column_to_plot)
plt.grid(True)
plt.show()

column_to_plot = 'Domain_Classification_Accuracy'
plt.figure(figsize=(10, 6))
plt.plot(df[column_to_plot])
plt.title(f'Line Plot of {column_to_plot}')
plt.xlabel('Index')
plt.ylabel(column_to_plot)
plt.grid(True)
plt.show()