import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import mode

try:
    from lightgbm import LGBMClassifier
    lgbm_available = True
except ImportError:
    lgbm_available = False
try:
    from catboost import CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False

train_df = pd.read_csv('Train_Data.csv')

# Rename columns to more understandable names, except 'age_group'
column_rename = {
    'SEQN': 'ID',
    'RIAGENDR': 'Gender',
    'PAQ605': 'PhysicalActivity',
    'BMXBMI': 'BMI',
    'LBXGLU': 'Glucose',
    'DIQ010': 'Diabetes',
    'LBXGLT': 'GlucoseTolerance',
    'LBXIN': 'Insulin'
    # 'age_group' is intentionally left unchanged
}
train_df.rename(columns=column_rename, inplace=True)

# Fill nulls in PhysicalActivity and Diabetes with 0 (as before)
train_df['PhysicalActivity'] = train_df['PhysicalActivity'].fillna(0)
train_df['Diabetes'] = train_df['Diabetes'].fillna(0)

# Show the number of NaN values in each column
print('Number of NaN values in each column:')
print(train_df.isna().sum())

# Drop rows where age_group is NaN
train_df = train_df.dropna(subset=['age_group'])

# Debug: print unique values before encoding
print('Unique values in age_group before encoding:', train_df['age_group'].unique())

# Standardize values to lowercase and strip spaces
train_df['age_group'] = train_df['age_group'].str.strip().str.lower()

# Ordinal encoding for age_group: 'adult' -> 0, 'senior' -> 1
train_df['age_group'] = train_df['age_group'].map({'adult': 0, 'senior': 1})

# Drop the ID column
train_df = train_df.drop(columns=['ID'])

# Show the number of NaN values in each column after dropping
print('\nAfter dropping rows with NaN in age_group, ordinal encoding, and dropping ID column:')
print(train_df.isna().sum())

# Impute NaN values for each column using XGBoost
for col in train_df.columns:
    if train_df[col].isna().sum() > 0:
        not_null = train_df[train_df[col].notna()]
        is_null = train_df[train_df[col].isna()]
        if not_null.shape[0] == 0 or is_null.shape[0] == 0:
            continue
        X = not_null.drop(columns=[col])
        y = not_null[col]
        X_pred = is_null.drop(columns=[col])
        # Use regression for numeric columns
        if np.issubdtype(y.dtype, np.number):
            X = X.fillna(-999)
            X_pred = X_pred.fillna(-999)
            model = XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X_pred)
            train_df.loc[train_df[col].isna(), col] = y_pred
        # Only use classification for categorical (object or int/float with few unique values)
        elif y.dtype == 'object' or (y.nunique() < 20):
            X = X.fillna(-999)
            X_pred = X_pred.fillna(-999)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            model.fit(X, y_encoded)
            y_pred = model.predict(X_pred)
            train_df.loc[train_df[col].isna(), col] = le.inverse_transform(y_pred)

# Show the DataFrame after XGBoost imputation
print('\nDataFrame after XGBoost classification imputation:')
with pd.option_context('display.max_columns', None):
    print(train_df.head(10))

# Show the number of NaN values in each column after XGBoost imputation
print('\nNumber of NaN values in each column after XGBoost imputation:')
print(train_df.isna().sum())

# Show the top 10 rows of the cleaned DataFrame
print('\nTop 10 rows of cleaned DataFrame:')
# Show all columns in the output
with pd.option_context('display.max_columns', None):
    print(train_df.head(10))

# Predict age_group using only XGBoost and print F1 score
X = train_df.drop(columns=['age_group'])
y = train_df['age_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, max_depth=2, reg_alpha=1.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'\nF1 Score (XGBoost, predicting age_group): {f1:.4f}')

# Evaluate XGBoost on both train and test data to check for overfitting
train_pred = model.predict(X_train)
f1_train = f1_score(y_train, train_pred, average='weighted')
print(f'F1 Score (XGBoost, training set): {f1_train:.4f}')
print(f'F1 Score (XGBoost, test set): {f1:.4f}')

# Train XGBoost on the full train_df
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=65, max_depth=2, reg_alpha=1.5)
X_full = train_df.drop(columns=['age_group'])
y_full = train_df['age_group']
model.fit(X_full, y_full)

# Load and preprocess the test data
submission_test = pd.read_csv('Test_Data (1).csv')
submission_test.rename(columns=column_rename, inplace=True)
# Fill missing columns if needed
for col in X_full.columns:
    if col not in submission_test.columns:
        submission_test[col] = 0
submission_test = submission_test[X_full.columns]  # Ensure same column order
submission_test = submission_test.fillna(-999)

# Predict age_group for the test data
age_group_pred = model.predict(submission_test)

# Save the result to submit.csv
result_df = pd.DataFrame({'age_group': age_group_pred})
result_df.to_csv('Ai-planet-submission.csv', index=False)
print('\nPredictions for Test_Data (1).csv saved to Ai-planet-submission.csv')


