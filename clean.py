import pandas as pd
from sklearn.impute import SimpleImputer

# Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Display the first few rows of the training data
print(train_df.head())

# Calculate missing value percentage for each column
missing_values = train_df.isnull().sum()
missing_percentage = (missing_values / len(train_df)) * 100

# Drop columns with more than 50% missing values
seuil = 0.5
cols_to_drop = train_df.columns[train_df.isnull().mean() > seuil]
train_df.drop(columns=cols_to_drop, inplace=True)
test_df.drop(columns=cols_to_drop, inplace=True)

# Identify columns with less than 10% missing values
cols_less_than_10_percent_missing = train_df.columns[train_df.isnull().mean() < 0.1]

# Impute numerical columns with the median
numerical_cols = train_df[cols_less_than_10_percent_missing].select_dtypes(include=['float64', 'int64']).columns
categorical_cols = train_df[cols_less_than_10_percent_missing].select_dtypes(include=['object']).columns

numerical_imputer = SimpleImputer(strategy='median')
train_df[numerical_cols] = numerical_imputer.fit_transform(train_df[numerical_cols])

# Impute categorical columns with the most frequent value (mode)
categorical_imputer = SimpleImputer(strategy='most_frequent')
train_df[categorical_cols] = categorical_imputer.fit_transform(train_df[categorical_cols])

# Check if any missing values remain
missing_data_after_imputation = train_df.isnull().sum().sort_values(ascending=False)
print(missing_data_after_imputation.head())

# Impute for specific columns (LotFrontage and FireplaceQu)
lotfrontage_imputer = SimpleImputer(strategy='median')
train_df['LotFrontage'] = lotfrontage_imputer.fit_transform(train_df[['LotFrontage']])

fireplacequ_imputer = SimpleImputer(strategy='most_frequent')
train_df['FireplaceQu'] = fireplacequ_imputer.fit_transform(train_df[['FireplaceQu']]).ravel()

# Check the missing values again to confirm the imputation
missing_values_after = train_df.isnull().sum()
print(missing_values_after)
