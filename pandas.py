import pandas as pd

# Step 1: Import data
file_path = '/Users/mukulhooda/Desktop/College/3rd Year/Machine Learning-1/Lab File/Programs/titanic/train.csv '
titanic_data = pd.read_csv(file_path)
print()
print("Original Data:") 
print(titanic_data.head()) 
print("\n")

#Data Exploration and Pre-processing 
# 1: Check for missing values 
print("Missing Values:") 
print(titanic_data.isnull().sum()) 
print("\n")

# 2: Handling Missing Values
titanic_data.dropna(subset=['Age'], inplace=True) 
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Display data after handling missing values 
print("Data after handling missing values:") 
print(titanic_data.head())
print("\n")

# 3: Removing Duplicates 
titanic_data.drop_duplicates(inplace=True)

# Data after removing duplicates 
print("Data after removing duplicates:") 
print(titanic_data.head())
print("\n")

# 4: Renaming Columns
titanic_data.rename(columns={'Pclass': 'PassengerClass', 'Survived': 'Survival'}, inplace=True)