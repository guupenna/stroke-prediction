import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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

numeric_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
        ]
    )

ct = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, ['age', 'avg_glucose_level', 'bmi']),
        ('cat', OneHotEncoder(), ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
    ],
    remainder='passthrough'
)

X_trans = ct.fit_transform(X_train)