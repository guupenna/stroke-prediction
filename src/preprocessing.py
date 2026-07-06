import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# loading csv
df_raw = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')

# dropping non impactful columns
df = df_raw.drop(columns=['id'])

# splitting dataset into train and test
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# copying and substituing dataframe columns with imputed values
imp = SimpleImputer(strategy='mean').fit(X_train[['bmi']])
X_imputed = imp.transform(X_train[['bmi']])
X_train_processed = X_train.copy()
X_train_processed[['bmi']] = X_imputed

# scaling numeric features
X_numeric = X_train_processed[['age', 'avg_glucose_level', 'bmi']]
scaler = preprocessing.StandardScaler().fit(X_numeric)
X_scaled = scaler.transform(X_numeric)

# encoding categoric variables
X_categoric = X_train[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
enc = preprocessing.OneHotEncoder().fit(X_categoric)
X_encoded = enc.transform(X_categoric)