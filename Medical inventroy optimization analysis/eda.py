# import pandas as pd
# import numpy as np
# import seaborn as sns

# data = pd.read_csv("E:/adeebs_lab_codes/analysis/Medical inventroy optimization analysis/Dataset/Medical-Inventory-Optimaization-Dataset.csv")

# #understanding the data
# data.head()
# data.tail()
# data.shape
# data.describe()
# data.columns
# data.nunique()

# # print(data.head())
# # print(data.tail())
# # print(data.shape)

# # cleaning

# data.isnull().sum()
# data.dropna(inplace=True)
# # n= data.drop(['',''],axis=1)

# # relationship analysis

# # co =n.corr()
# # sns.heatmap(corelatoin, xticklabels=co.columns, yticklabels=co.columns,annot=True)
# # sns.pairplot(n)
# # sns.relplot(x='',y='',hue='',data=n)
# # sns.distplat(n[''],bins=10)
# # sns.catplot(x='',kind='box',data=n)

# import pandas as pd

# # Load the uploaded Excel file to inspect the sheets and data structure
# file_path = 'E:/adeebs_lab_codes/analysis/Medical inventroy optimization analysis/Dataset/Medical Inventory Optimaization Dataset.xlsx'
# xls = pd.ExcelFile(file_path)

# # Check the sheet names to understand the data structure
# xls.sheet_names
# # Load the main sheet with the project data to inspect the first few rows and understand the columns
# df = pd.read_excel(xls, sheet_name='Projectfinaldata (1)')
# df.head()


# # Convert date fields to proper datetime format
# df['Dateofbill'] = pd.to_datetime(df['Dateofbill'], errors='coerce')

# # Check for missing values in the dataset
# missing_values = df.isnull().sum()

# # Summary statistics for numeric columns
# summary_stats = df.describe()

# # Check the unique values for categorical columns like 'Specialisation', 'Dept', 'DrugName'
# unique_specialisations = df['Specialisation'].unique()
# unique_depts = df['Dept'].unique()
# unique_drugs = df['DrugName'].unique()

# missing_values, summary_stats, unique_specialisations, unique_depts, unique_drugs


# # Dropping rows with missing 'Final_Sales' or 'Final_Cost' as they are critical for analysis
# df_cleaned = df.dropna(subset=['Final_Sales', 'Final_Cost'])

# # Removing duplicates
# df_cleaned = df_cleaned.drop_duplicates()

# # Handling outliers by capping extreme values for 'Final_Sales', 'ReturnQuantity'
# df_cleaned['Final_Sales'] = df_cleaned['Final_Sales'].clip(lower=0)
# df_cleaned['ReturnQuantity'] = df_cleaned['ReturnQuantity'].clip(lower=0)


# # Filter data for sales that had returns
# df_filtered = df_cleaned[df_cleaned['ReturnQuantity'] > 0]

# # Alternatively, filter by a specific department or date range if required
# recent_sales = df_cleaned[df_cleaned['Dateofbill'] >= '2023-01-01']  # Example filter for recent data


# # Calculate the Return Rate
# df_cleaned['ReturnRate'] = df_cleaned['ReturnQuantity'] / df_cleaned['Quantity']

# # Calculate the Profit
# df_cleaned['Profit'] = df_cleaned['Final_Sales'] - df_cleaned['Final_Cost']

# # Grouping data by Specialization, Department, DrugName for analysis
# grouped_data = df_cleaned.groupby(['Specialisation', 'Dept', 'DrugName']).agg({
#     'Final_Sales': 'sum',
#     'Final_Cost': 'sum',
#     'ReturnQuantity': 'sum',
#     'Profit': 'sum'
# }).reset_index()

# grouped_data['ReturnRate'] = grouped_data['ReturnQuantity'] / grouped_data['Final_Sales']


# # Redefine the dataframe and run the save process again as the kernel has reset.

# # Save the cleaned and preprocessed data into a new Excel file for use in Power BI
# output_file_path = 'E:/adeebs_lab_codes/analysis/Medical inventroy optimization analysis/Dataset/Medical_Inventory_Cleaned_Preprocessed.xlsx'
# df_cleaned.to_excel(output_file_path, index=False)

# # Also save the grouped data (aggregated by Specialisation, Dept, and DrugName) into a separate sheet
# with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
#     df_cleaned.to_excel(writer, sheet_name='CleanedData', index=False)
#     grouped_data.to_excel(writer, sheet_name='GroupedData', index=False)

# # Return the path to the file for download
# output_file_path


# import pandas as pd

# # Reload the Excel file
# file_path = 'E:/adeebs_lab_codes/analysis/Medical inventroy optimization analysis/Dataset/Medical Inventory Optimaization Dataset.xlsx'
# xls = pd.ExcelFile(file_path)

# # Load the main sheet to preprocess
# df = pd.read_excel(xls, sheet_name='Projectfinaldata (1)')

# # Perform the cleaning and preprocessing steps again
# df['Dateofbill'] = pd.to_datetime(df['Dateofbill'], errors='coerce')

# # Clean the data
# df_cleaned = df.dropna(subset=['Final_Sales', 'Final_Cost'])
# df_cleaned = df_cleaned.drop_duplicates()
# df_cleaned['Final_Sales'] = df_cleaned['Final_Sales'].clip(lower=0)
# df_cleaned['ReturnQuantity'] = df_cleaned['ReturnQuantity'].clip(lower=0)

# # Preprocess - Calculating new metrics
# df_cleaned['ReturnRate'] = df_cleaned['ReturnQuantity'] / df_cleaned['Quantity']
# df_cleaned['Profit'] = df_cleaned['Final_Sales'] - df_cleaned['Final_Cost']

# # Grouping the data for further analysis
# grouped_data = df_cleaned.groupby(['Specialisation', 'Dept', 'DrugName']).agg({
#     'Final_Sales': 'sum',
#     'Final_Cost': 'sum',
#     'ReturnQuantity': 'sum',
#     'Profit': 'sum'
# }).reset_index()
# grouped_data['ReturnRate'] = grouped_data['ReturnQuantity'] / grouped_data['Final_Sales']

# # Save the cleaned and preprocessed data into a new Excel file for Power BI visualization
# output_file_path = '/mnt/data/Medical_Inventory_Cleaned_Preprocessed.xlsx'
# with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
#     df_cleaned.to_excel(writer, sheet_name='CleanedData', index=False)
#     grouped_data.to_excel(writer, sheet_name='GroupedData', index=False)

# # Provide the download path
# output_file_path


# # Remove rows with null values in the columns 'Formulation', 'DrugName', 'SubCat', and 'SubCat1'
# df_cleaned_filtered = df_cleaned.dropna(subset=['Formulation', 'DrugName', 'SubCat', 'SubCat1'])

# # Replace 'inf' values in the ReturnRate column with 0
# df_cleaned_filtered['ReturnRate'] = df_cleaned_filtered['ReturnRate'].replace([float('inf'), -float('inf')], 0)

# # Save the cleaned and filtered data into a new Excel file for Power BI visualization
# output_file_path_filtered = '/mnt/data/Medical_Inventory_Cleaned_Filtered_Preprocessed.xlsx'
# with pd.ExcelWriter(output_file_path_filtered, engine='xlsxwriter') as writer:
#     df_cleaned_filtered.to_excel(writer, sheet_name='CleanedFilteredData', index=False)
#     grouped_data.to_excel(writer, sheet_name='GroupedData', index=False)

# # Provide the download path
# output_file_path_filtered


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "E:/adeebs_lab_codes/analysis/Medical inventroy optimization analysis/Dataset/Medical Inventory Optimaization Dataset.xlsx"
df = pd.read_excel(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Checking for missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# Filling missing values (example: forward fill for time-series data, mean for numerical)
df.fillna(method='ffill', inplace=True)  # Modify as per dataset needs

# Checking the distribution of stock levels
plt.figure(figsize=(12,6))
sns.histplot(df['Quantity'], bins=30, kde=True, color='blue')
plt.title("Distribution of Stock Levels")
plt.xlabel("Stock Level")
plt.ylabel("Frequency")
plt.show()

# Category-wise stock distribution
plt.figure(figsize=(14,7))
sns.boxplot(x='SubCat', y='Quantity', data=df)
plt.xticks(rotation=45)
plt.title("Stock Levels by Category 1")
plt.show()

# # Category-wise stock distribution
plt.figure(figsize=(14,7))
sns.boxplot(x='SubCat1', y='Quantity', data=df)
plt.xticks(rotation=45)
plt.title("Stock Levels by Category ")
plt.show()

# Monthly inventory trends (if date column exists)
# if 'Date' in df.columns:
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'])
df.set_index('Dateofbill', inplace=True)
plt.figure(figsize=(14,6))
df['Quantity'].resample('M').sum().plot(marker='o', linestyle='-')
plt.title("Monthly Stock Level Trends")
plt.ylabel("Stock Level")
plt.show()

# Display column data types
print(df.dtypes)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Check if numeric_df is empty (in case all columns are non-numeric)
if numeric_df.empty:
    print("No numeric columns found in the dataset. Please check the data.")
else:
    # Plot the heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Insights & Optimization Recommendations
print("\nOptimization Recommendations:")
print("1. Identify slow-moving and fast-moving inventory items.")
print("2. Set reorder points based on consumption trends.")
print("3. Reduce overstocking by forecasting demand accurately.")
print("4. Implement Just-In-Time (JIT) inventory strategy to reduce waste.")
print("5. Monitor stock levels and restocking efficiency regularly.")
