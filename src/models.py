from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

def create_knn(preprocessor):
    return Pipeline(
        steps=[
            ('preprocessing', preprocessor),
            ('knn', KNeighborsClassifier())
        ]
    )