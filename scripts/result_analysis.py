import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the CSV file into a pandas DataFrame
# Replace 'your_file.csv' with the path to your actual CSV file
file_path = './concatenated_dataframe.csv'
# file_path = './results/concatenated_dataframe_28_05.csv'
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
indices = [np.where(df['Step number'] == step)[0][0] for step in unique_step_numbers]

unique_mae_loss = df['Generator mae loss'].unique()
unique_domain_acc = df['Domain_Classification_Accuracy'].unique()
print(unique_step_numbers[-100:])
print(unique_mae_loss[-100:])


#columns_to_plot = ['Domain_Classification_Accuracy','Dicriminator loss', 'Class accuracy', 'Label loss', 'Generator class loss']
columns_to_plot = df.columns 

for column_to_plot in columns_to_plot:
    #column_to_plot = 'Domain_Classification_Accuracy'
    plt.figure(figsize=(10, 6))
    plt.plot(unique_step_numbers, df[column_to_plot].iloc[indices])
    #plt.plot(df[column_to_plot])
    plt.title(f'Line Plot of {column_to_plot}')
    plt.xlabel('Index')
    plt.ylabel(column_to_plot)
    plt.grid(True)
    plt.show()

plt.plot(unique_step_numbers, df['Disc real loss'].iloc[indices], label = 'Real')
plt.plot(unique_step_numbers, df['Disc fake loss'].iloc[indices], label = 'Fake')
plt.plot(unique_step_numbers,  df['Disc real loss'].iloc[indices] +  df['Disc real loss'].iloc[indices], label = "Combined")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(unique_step_numbers, df['Disc category loss real'].iloc[indices], label = 'Real')
plt.plot(unique_step_numbers, df['Disc category loss fake'].iloc[indices], label = 'Fake')
plt.plot(unique_step_numbers,  df['Disc category loss real'].iloc[indices] +  df['Disc category loss fake'].iloc[indices], label = "Combined")
plt.grid(True)
plt.legend()
plt.show()

