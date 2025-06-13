import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis

train_df = pd.read_csv("hacktrain.csv")
test_df = pd.read_csv("hacktest.csv")

ndvi_cols = [col for col in train_df.columns if col.endswith('_N')]

train_df[ndvi_cols] = train_df[ndvi_cols].fillna(train_df[ndvi_cols].mean())
test_df[ndvi_cols] = test_df[ndvi_cols].fillna(test_df[ndvi_cols].mean())

def extract_advanced_features(df):
    features = pd.DataFrame()
    features['ID'] = df['ID']
    ndvi = df[ndvi_cols].values

    features['mean_ndvi'] = np.mean(ndvi, axis=1)
    features['std_ndvi'] = np.std(ndvi, axis=1)
    features['max_ndvi'] = np.max(ndvi, axis=1)
    features['min_ndvi'] = np.min(ndvi, axis=1)
    features['range_ndvi'] = features['max_ndvi'] - features['min_ndvi']
    features['median_ndvi'] = np.median(ndvi, axis=1)
    features['skew_ndvi'] = skew(ndvi, axis=1)
    features['kurt_ndvi'] = kurtosis(ndvi, axis=1)

    diffs = np.diff(ndvi, axis=1)
    features['mean_diff'] = np.mean(diffs, axis=1)
    features['std_diff'] = np.std(diffs, axis=1)
    features['min_diff'] = np.min(diffs, axis=1)
    features['max_diff'] = np.max(diffs, axis=1)
    features['sum_diff'] = np.sum(diffs, axis=1)

    features['incr_steps'] = (diffs > 0).sum(axis=1)
    features['decr_steps'] = (diffs < 0).sum(axis=1)

    features['change_start_end'] = ndvi[:, -1] - ndvi[:, 0]

    features['idx_max_ndvi'] = np.argmax(ndvi, axis=1)
    features['idx_min_ndvi'] = np.argmin(ndvi, axis=1)

    return features

X_train = extract_advanced_features(train_df)
X_test = extract_advanced_features(test_df)

le = LabelEncoder()
y_train = le.fit_transform(train_df['class'])

model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000, random_state=42)
model.fit(X_train.drop(columns=['ID']), y_train)

preds = model.predict(X_test.drop(columns=['ID']))
pred_labels = le.inverse_transform(preds)

submission = pd.DataFrame({
    'ID': X_test['ID'],
    'class': pred_labels
})

submission.to_csv("submission.csv", index=False)
print("Enhanced submission file saved as submission.csv")
