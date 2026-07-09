import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    # loading csv
    df_raw = pd.read_csv('../data/healthcare-dataset-stroke-data.csv')

    # dropping non impactful columns
    df = df_raw.drop(columns=['id'])

    # splitting dataset into train and test
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    return train_test_split(X, y, test_size=0.25, random_state=42)


def create_preprocessor():
    # creating pipeline for dealing with numeric features
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]
    )

    # creating column transformer for aggregate transformer operations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, ['age', 'avg_glucose_level', 'bmi']),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
        ],
        remainder='passthrough'
    )

    return preprocessor